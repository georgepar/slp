import math

import torch.nn as nn

from slp.modules.attention import MultiheadAttention
from slp.modules.embed import PositionalEncoding, Embed
from slp.modules.feedforward import PositionwiseFF
from slp.modules.norm import LayerNorm
from slp.util.pytorch import repeat_layer


class Sublayer1(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8, dropout=0.1):
        super(Sublayer1, self).__init__()
        self.lnorm = LayerNorm(hidden_size)
        self.sublayer = MultiheadAttention(
            attention_size=hidden_size, num_heads=num_heads, dropout=dropout
        )

    def forward(self, x, attention_mask=None):
        return self.lnorm(x + self.sublayer(x, attention_mask=attention_mask))


class Sublayer2(nn.Module):
    def __init__(self, hidden_size=512, inner_size=2048, dropout=0.1):
        super(Sublayer2, self).__init__()
        self.lnorm = LayerNorm(hidden_size)
        self.sublayer = PositionwiseFF(hidden_size, inner_size, dropout=dropout)

    def forward(self, x):
        return self.lnorm(x + self.sublayer(x))


class Sublayer3(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8, dropout=0.1):
        super(Sublayer3, self).__init__()
        self.lnorm = LayerNorm(hidden_size)
        self.sublayer = MultiheadAttention(
            attention_size=hidden_size, num_heads=num_heads, dropout=dropout
        )

    def forward(self, x, y, attention_mask=None):
        return self.lnorm(
            x + self.sublayer(x, queries=y, values=x, attention_mask=attention_mask)
        )


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8, inner_size=2048, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.l1 = Sublayer1(
            hidden_size=hidden_size, num_heads=num_heads, dropout=dropout
        )
        self.l2 = Sublayer2(
            hidden_size=hidden_size, inner_size=inner_size, dropout=dropout
        )

    def forward(self, x, attention_mask=None):
        out = self.l1(x, attention_mask=attention_mask)
        out = self.l2(out)
        return out


class Encoder(nn.Module):
    def __init__(
        self, num_layers=6, hidden_size=512, num_heads=8, inner_size=2048, dropout=0.1
    ):
        super(Encoder, self).__init__()
        self.encoder = nn.ModuleList(
            repeat_layer(
                EncoderLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    inner_size=inner_size,
                    dropout=dropout,
                ),
                num_layers,
            )
        )

    def forward(self, x, attention_mask=None):
        for layer in self.encoder:
            x = layer(x, attention_mask=attention_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8, inner_size=2048, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.in_layer = Sublayer1(
            hidden_size=hidden_size, num_heads=num_heads, dropout=dropout
        )
        self.fuse_layer = Sublayer3(
            hidden_size=hidden_size, num_heads=num_heads, dropout=dropout
        )
        self.out_layer = Sublayer2(
            hidden_size=hidden_size, inner_size=inner_size, dropout=dropout
        )

    def forward(self, targets, encoded, source_mask=None, target_mask=None):
        targets = self.in_layer(targets, attention_mask=target_mask)
        out = self.fuse_layer(encoded, targets, attention_mask=source_mask)
        out = self.out_layer(out)
        return out


class Decoder(nn.Module):
    def __init__(
        self, num_layers=6, hidden_size=512, num_heads=8, inner_size=2048, dropout=0.1
    ):
        super(Decoder, self).__init__()
        self.decoder = nn.ModuleList(
            repeat_layer(
                DecoderLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    inner_size=inner_size,
                    dropout=dropout,
                ),
                num_layers,
            )
        )

    def forward(self, target, encoded, source_mask=None, target_mask=None):

        for l in self.decoder:
            target = l(
                target, encoded, source_mask=source_mask, target_mask=target_mask
            )
        return target


class EncoderDecoder(nn.Module):
    def __init__(
        self, num_layers=6, hidden_size=512, num_heads=8, inner_size=2048, dropout=0.1
    ):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            inner_size=inner_size,
            dropout=dropout,
        )
        self.decoder = Decoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            inner_size=inner_size,
            dropout=dropout,
        )

    def forward(self, source, target, source_mask=None, target_mask=None):
        encoded = self.encoder(source, attention_mask=source_mask)
        decoded = self.decoder(
            target, encoded, source_mask=source_mask, target_mask=target_mask
        )
        return decoded


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size=30000,
        max_length=256,
        num_layers=6,
        hidden_size=512,
        num_heads=8,
        inner_size=2048,
        dropout=0.1,
    ):
        super(Transformer, self).__init__()
        self.embed = Embed(
            vocab_size,
            hidden_size,
            scale=math.sqrt(hidden_size),
            dropout=dropout,
            trainable=True,
        )
        self.pe = PositionalEncoding(embedding_dim=hidden_size, max_len=max_length)
        self.transformer_block = EncoderDecoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            inner_size=inner_size,
            dropout=dropout,
        )
        self.drop = nn.Dropout(dropout)
        self.predict = nn.Linear(hidden_size, vocab_size)
        self._reset_parameters()

    def forward(self, source, target, source_mask=None, target_mask=None):
        source = self.embed(source)
        target = self.embed(target)
        # Adding embeddings + pos embeddings
        # is done in PositionalEncoding class
        source = self.pe(source)
        target = self.pe(target)
        out = self.transformer_block(
            source, target, source_mask=source_mask, target_mask=target_mask
        )
        out = self.drop(out)
        out = self.predict(out)
        return out

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
