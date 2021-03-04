import itertools
import torch
import torch.nn as nn

from typing import Union, List

from loguru import logger
from slp.modules.norm import LayerNorm


# Activation functions to choose
NON_LINEARITIES = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "none": None,
}


class FF(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        activation: str = "relu",
        layer_norm: bool = True,
        bias: bool = True,
        dropout: float = 0.1,
    ):
        """Feed-forward layer with optional layernorm, and activation

        Args:
            n_in (int): Input features
            n_out (int): Output features
            activation (str): Which activation to use. Defaults to "relu".
            layer_norm (bool): Use layernorm. Defaults to True.
            bias (bool): Add bias. Defaults to True.
            dropout (float): Dropout probability. Defaults to 0.1.
        """
        super(FF, self).__init__()
        self.fc = nn.Linear(n_in, n_out, bias=bias)
        self.activation = NON_LINEARITIES.get(activation, nn.ReLU)
        if self.activation is not None:
            self.activation = self.activation()
        self.layer_norm = None
        if layer_norm:
            self.layer_norm = LayerNorm(n_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward FF pass

        Args:
            x (torch.Tensor): [B, *, D] input features

        Returns:
            torch.Tensor: [B, *, D] output features
        """
        out = self.fc(x)
        out = self.drop(out)
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        if self.activation is not None:
            out = self.activation(out)
        return out  # type: ignore


class PositionwiseFF(nn.Module):
    """Some Information about PositionwiseFF"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """Transformer Position-wise feed-forward layer

        Linear -> LayerNorm -> ReLU -> Linear

        Args:
            d_model (int): Model dimension
            d_ff (int): Hidden dimension
            dropout (float): Dropout probability. Defaults to 0.1.
        """
        super(PositionwiseFF, self).__init__()
        self.ff1 = FF(d_model, d_ff, activation="relu")
        self.ff2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
        self.net = nn.Sequential(self.ff1, self.drop, self.ff2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Position-wise FF forward pass

        $$out = W_2 \dot max(0, W_1 \dot x + b_1) + b_2$$

        [B, *, D] -> [B, *, H] -> [B, *, D]

        * B: Batch size
        * D: Model dim
        * H: Hidden size > Model dim (Usually $H = 2D$)

        Args:
            x (torch.Tensor): [B, *, D] Input features

        Returns:
            torch.Tensor: [B, *, D] Output features
        """
        out: torch.Tensor = self.net(x)
        return out


class MultilayerFF(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        sizes: Union[int, List[int]],
        n_layers: int = 1,
        activation: str = "relu",
        dropout: float = 0.1,
    ):
        """Multi-layer feed forward layer.

        Stacks multiple FFs with configurable dimensions

        Args:
            input_dim (int): Input features
            output_dim (int): Output features
            sizes (Union[int, List[int]]): Intermediate layer features as a list of dimensions.
                If single int is passed all intermediate layers have the same dimension
            n_layers (int): Number of intermediate layers. Defaults to 1.
            activation (str): Activation function. Defaults to "relu".
            dropout (float): Dropout probability. Defaults to 0.1.
        """
        super(MultilayerFF, self).__init__()
        if isinstance(sizes, int):
            sizes = list(itertools.repeat(sizes, n_layers))  # [n] * l
        sizes = [input_dim] + sizes + [output_dim]
        if len(sizes) != n_layers + 2:
            logger.warning(
                f"n_layers={n_layers} does not match len of "
                "sizes={len(sizes)}. Using {len(sizes)} layers"
            )
        self.net = nn.Sequential(
            *[
                FF(nin, nout, activation=activation, dropout=dropout)
                for nin, nout in zip(sizes[:-1], sizes[1:])
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward Multilayer-FF pass

        Args:
            x (torch.Tensor): [B, *, D] input features

        Returns:
            torch.Tensor: [B, *, D] output features
        """
        out: torch.Tensor = self.net(x)
        return out
