import itertools
import torch.nn as nn

from slp.util import log
from slp.modules.norm import LayerNorm

NON_LINEARITIES = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'none': None
}


class FF(nn.Module):
    """Some Information about FF"""
    def __init__(self,
                 n_in, n_out, activation='relu',
                 layer_norm=True, bias=True, dropout=.1):
        super(FF, self).__init__()
        self.fc = nn.Linear(n_in, n_out, bias=bias)
        self.activation = NON_LINEARITIES.get(activation, nn.ReLU)
        if self.activation is not None:
            self.activation = self.activation()
        self.layer_norm = None
        if layer_norm:
            self.layer_norm = LayerNorm(n_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.fc(x)
        out = self.drop(out)
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        if self.activation is not None:
            out = self.activation(out)
        return out


class PositionwiseFF(nn.Module):
    """Some Information about PositionwiseFF"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFF, self).__init__()
        self.ff1 = FF(d_model, d_ff, activation='relu')
        self.ff2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
        self.net = nn.Sequential(self.ff1, self.drop, self.ff2)

    def forward(self, x):
        # out = max(0, W1 * x + b1) * W2 + b2
        return self.net(x)


class MultilayerFF(nn.Module):
    """Some Information about MLFF"""
    def __init__(self,
                 input_dim,
                 output_dim,
                 sizes,
                 n_layers=1,
                 activation='relu',
                 dropout=.1):
        super(MultilayerFF, self).__init__()
        if isinstance(sizes, int):
            sizes = list(itertools.repeat(sizes, n_layers))  # [n] * l
        sizes = [input_dim] + sizes + [output_dim]
        if len(sizes) != n_layers + 2:
            log.warn(f'n_layers={n_layers} does not match len of '
                     'sizes={len(sizes)}. Using {len(sizes)} layers')
        self.net = nn.Sequential(*[
            FF(nin, nout, activation=activation, dropout=dropout)
            for nin, nout in zip(sizes[:-1], sizes[1:])])

    def forward(self, x):
        return self.net(x)
