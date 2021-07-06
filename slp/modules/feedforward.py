import itertools
from typing import List, Union

import torch
import torch.nn as nn
from loguru import logger
from slp.modules.norm import LayerNorm
from slp.util.pytorch import NoOp

# Activation functions to choose
NON_LINEARITIES = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "none": NoOp,
}


class TwoLayer(nn.Module):
    def __init__(
        self,
        n_in: int,
        inner_dim: int,
        n_out: int,
        activation: str = "relu",
        bias: bool = True,
        dropout: float = 0.1,
        residual: bool = False,
    ):
        super(TwoLayer, self).__init__()
        self.l1 = nn.Linear(n_in, inner_dim)
        self.act = (
            NON_LINEARITIES[activation]()
            if activation in NON_LINEARITIES
            else nn.ReLU()
        )
        self.l2 = nn.Linear(inner_dim, n_out)
        self.drop = nn.Dropout(p=dropout)

        if residual:
            assert n_in == n_out, "Residual connection assumes n_in == n_out"
        self.residual = residual

    def forward(self, x):
        out = self.l1(x)
        out = self.drop(out)
        out = self.act(out)
        out = self.l2(out)
        out = self.drop(out)

        if self.residual:
            out = x + out

        return out


class PositionwiseFF(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, gelu=False):
        """Transformer Position-wise feed-forward layer

        Linear -> LayerNorm -> ReLU -> Linear

        Args:
            d_model (int): Model dimension
            d_ff (int): Hidden dimension
            dropout (float): Dropout probability. Defaults to 0.1.
        """
        super(PositionwiseFF, self).__init__()
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
        self.activation = nn.ReLU() if not gelu else nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Position-wise FF forward pass

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
        out: torch.Tensor = self.ff2(self.drop(self.activation(self.ff1(x))))
        return out
