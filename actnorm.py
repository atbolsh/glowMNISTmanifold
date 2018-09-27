"""
Normalization to improve stability.
Ripped from GLOW.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.linalg


import thops


class ActNorm(nn.Module):
    """
    Activation Normalization
    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance for that.

    After initialization, `bias` and `logs` will be trained as parameters.
    """

    def __init__(self, num_features, pixelDims = 0, scale=1.):
        super().__init__()
        # register mean and scale
        self.pixelDims = pixelDims
        size = [1 for i in range(pixelDims + 2)]
        size[1] = num_features
        self.non_feature_dims = [0] + list(range(2, 2+pixelDims))

        self.register_parameter("bias", nn.Parameter(torch.zeros(*size)))
        self.register_parameter("logs", nn.Parameter(torch.zeros(*size)))
        self.num_features = num_features
        self.scale = float(scale)
        self.inited = False

    def _check_input_dim(self, input):
        assert len(input.size()) == self.pixelDims + 2
        assert input.size(1) == self.num_features, (
            "[ActNorm]: input should be in shape as `BC(pixel dimensions)`,"
            " channels should be {} rather than {}".format(
                self.num_features, input.size()))

    def initialize_parameters(self, input):
        self._check_input_dim(input)
        if not self.training:
            return
        assert input.device == self.bias.device
        with torch.no_grad():
            bias = thops.mean(input.clone(), dim=self.non_feature_dims, keepdim=True) * -1.0
            vars = thops.mean((input.clone() + bias) ** 2, dim=self.non_feature_dims, keepdim=True)
            logs = torch.log(self.scale/(torch.sqrt(vars)+1e-6))
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.inited = True

    def _center(self, input, reverse=False):
        if not reverse:
            return input + self.bias
        else:
            return input - self.bias

    def _scale(self, input, reverse=False):
        logs = self.logs
        if not reverse:
            input = input * torch.exp(logs)
        else:
            input = input * torch.exp(-logs)
        logdet = thops.sum(logs) * thops.pixels(input)
#        if reverse:
#            logdet *= -1
        return input, logdet

    def forward(self, input, reverse=False):
        if not self.inited:
            self.initialize_parameters(input)
        self._check_input_dim(input)
        # no need to permute dims as old version
        if not reverse:
            # center and scale
            input = self._center(input, reverse)
            input, logdet = self._scale(input, reverse)
        else:
            # scale and center
            input, logdet = self._scale(input, reverse)
            input = self._center(input, reverse)
        return input, logdet

    def pushback(self, input):
        with torch.no_grad():
            return self.forward(input, reverse=True)


class ActNorm2d(ActNorm):
    def __init__(self, num_features, scale=1.):
        super().__init__(num_features, scale)

    def _check_input_dim(self, input):
        assert len(input.size()) == 4
        assert input.size(1) == self.num_features, (
            "[ActNorm]: input should be in shape as `BCHW`,"
            " channels should be {} rather than {}".format(
                self.num_features, input.size()))


