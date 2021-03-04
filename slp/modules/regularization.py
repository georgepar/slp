import torch
from torch import nn
from torch.autograd import Variable


class GaussianNoise(nn.Module):
    def __init__(self, stddev: float, mean: float = 0.0):
        """Additive Gaussian Noise layer

        Args:
            stddev (float): the standard deviation of the distribution
            mean (float): the mean of the distribution
        """
        super().__init__()
        self.stddev = stddev
        self.mean = mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gaussian noise forward pass

        Args:
            x (torch.Tensor): Input features.

        Returns:
            [type]: [description]
        """
        if self.training:
            noise = Variable(x.data.new(x.size()).normal_(self.mean, self.stddev))
            return x + noise
        return x

    def __repr__(self):
        """String representation of class"""
        return "{} (mean={}, stddev={})".format(
            self.__class__.__name__, str(self.mean), str(self.stddev)
        )
