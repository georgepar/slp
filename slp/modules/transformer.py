import math

import torch.nn as nn
from slp.modules.attention import MultiheadAttention
from slp.modules.embed import Embed, PositionalEncoding
from slp.modules.feedforward import PositionwiseFF
from slp.modules.norm import LayerNorm, ScaleNorm
from slp.util.pytorch import repeat_layer


def reset_parameters(named_parameters, gain=1.0):
    """Initialize parameters in the transformer model."""

    for name, p in named_parameters:
        if p.dim() > 1:
            if "weight" in name:
                nn.init.xavier_normal_(p, gain=gain)

            if "bias" in name:
                nn.init.constant_(p, 0.0)


class Sublayer1(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        num_heads=8,
        dropout=0.1,
        nystrom=False,
        num_landmarks=32,
        kernel_size=None,
        prenorm=True,
        scalenorm=True,
    ):
        super(Sublayer1, self).__init__()
        self.sublayer = MultiheadAttention(
            attention_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            nystrom=nystrom,
            kernel_size=kernel_size,
            num_landmarks=num_landmarks,
        )
        self.prenorm = prenorm
        self.lnorm = LayerNorm(hidden_size) if not scalenorm else ScaleNorm(hidden_size)

    def _prenorm(self, x, attention_mask=None):
        out, _ = self.sublayer(self.lnorm(x), attention_mask=attention_mask)

        return out + x

    def _postnorm(self, x, attention_mask=None):
        out, _ = self.sublayer(x, attention_mask=attention_mask)

        return self.lnorm(x + out)

    def forward(self, x, attention_mask=None):
        return (
            self._prenorm(x, attention_mask=attention_mask)
            if self.prenorm
            else self._postnorm(x, attention_mask=attention_mask)
        )


class Sublayer2(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        inner_size=2048,
        dropout=0.1,
        prenorm=True,
        scalenorm=True,
    ):
        super(Sublayer2, self).__init__()
        self.sublayer = PositionwiseFF(hidden_size, inner_size, dropout=dropout)
        self.prenorm = prenorm
        self.lnorm = LayerNorm(hidden_size) if not scalenorm else ScaleNorm(hidden_size)

    def _prenorm(self, x):
        out = self.sublayer(self.lnorm(x))

        return out + x

    def _postnorm(self, x):
        out = self.sublayer(x)

        return self.lnorm(x + out)

    def forward(self, x):
        return self._prenorm(x) if self.prenorm else self._postnorm(x)


class Sublayer3(nn.Module):
    def __init__(
        self, hidden_size=512, num_heads=8, dropout=0.1, prenorm=True, scalenorm=True
    ):
        super(Sublayer3, self).__init__()
        self.sublayer = MultiheadAttention(
            attention_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            nystrom=False,  # Nystrom used only for self-attention
            kernel_size=None,  # convolutional residual not used when subsequent mask
        )
        self.prenorm = prenorm
        self.lnorm = LayerNorm(hidden_size) if not scalenorm else ScaleNorm(hidden_size)

        if self.prenorm:
            self.lnormy = (
                LayerNorm(hidden_size) if not scalenorm else ScaleNorm(hidden_size)
            )

    def _prenorm(self, x, y, attention_mask=None):
        out, _ = self.sublayer(
            self.lnorm(x), queries=self.lnormy(y), attention_mask=attention_mask
        )

        return out + x

    def _postnorm(self, x, y, attention_mask=None):
        out, _ = self.sublayer(x, queries=y, attention_mask=attention_mask)

        return self.lnorm(x + out)

    def forward(self, x, y, attention_mask=None):
        return (
            self._prenorm(x, y, attention_mask=attention_mask)
            if self.prenorm
            else self._postnorm(x, y, attention_mask=attention_mask)
        )


class EncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        num_heads=8,
        inner_size=2048,
        dropout=0.1,
        nystrom=False,
        num_landmarks=32,
        kernel_size=None,
        prenorm=True,
        scalenorm=True,
    ):
        super(EncoderLayer, self).__init__()
        self.l1 = Sublayer1(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            nystrom=nystrom,
            num_landmarks=num_landmarks,
            kernel_size=kernel_size,
            prenorm=prenorm,
            scalenorm=scalenorm,
        )
        self.l2 = Sublayer2(
            hidden_size=hidden_size,
            inner_size=inner_size,
            dropout=dropout,
            prenorm=prenorm,
            scalenorm=scalenorm,
        )

    def forward(self, x, attention_mask=None):
        out = self.l1(x, attention_mask=attention_mask)
        out = self.l2(out)

        return out


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers=6,
        hidden_size=512,
        num_heads=8,
        inner_size=2048,
        dropout=0.1,
        nystrom=False,
        num_landmarks=32,
        kernel_size=None,
        prenorm=True,
        scalenorm=True,
    ):
        super(Encoder, self).__init__()
        self.encoder = nn.ModuleList(
            repeat_layer(
                EncoderLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    inner_size=inner_size,
                    dropout=dropout,
                    nystrom=nystrom,
                    num_landmarks=num_landmarks,
                    kernel_size=kernel_size,
                    prenorm=prenorm,
                    scalenorm=scalenorm,
                ),
                num_layers,
            )
        )

    def forward(self, x, attention_mask=None):
        for layer in self.encoder:
            x = layer(x, attention_mask=attention_mask)

        return x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        num_heads=8,
        inner_size=2048,
        dropout=0.1,
        prenorm=True,
        scalenorm=True,
    ):
        super(DecoderLayer, self).__init__()
        self.in_layer = Sublayer1(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            nystrom=False,
            kernel_size=None,
            prenorm=prenorm,
            scalenorm=scalenorm,
        )
        self.fuse_layer = Sublayer3(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            prenorm=prenorm,
            scalenorm=scalenorm,
        )
        self.out_layer = Sublayer2(
            hidden_size=hidden_size,
            inner_size=inner_size,
            dropout=dropout,
            prenorm=prenorm,
            scalenorm=scalenorm,
        )

    def forward(self, targets, encoded, source_mask=None, target_mask=None):
        targets = self.in_layer(targets, attention_mask=target_mask)
        out = self.fuse_layer(encoded, targets, attention_mask=source_mask)
        out = self.out_layer(out)

        return out


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers=6,
        hidden_size=512,
        num_heads=8,
        inner_size=2048,
        dropout=0.1,
        prenorm=True,
        scalenorm=True,
    ):
        super(Decoder, self).__init__()
        self.decoder = nn.ModuleList(
            repeat_layer(
                DecoderLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    inner_size=inner_size,
                    dropout=dropout,
                    prenorm=prenorm,
                    scalenorm=scalenorm,
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
        self,
        num_layers=6,
        hidden_size=512,
        num_heads=8,
        inner_size=2048,
        dropout=0.1,
        nystrom=False,
        num_landmarks=32,
        kernel_size=None,
        prenorm=True,
        scalenorm=True,
    ):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            inner_size=inner_size,
            dropout=dropout,
            nystrom=nystrom,
            num_landmarks=num_landmarks,
            kernel_size=kernel_size,
            prenorm=prenorm,
            scalenorm=scalenorm,
        )
        self.decoder = Decoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            inner_size=inner_size,
            dropout=dropout,
            prenorm=prenorm,
            scalenorm=scalenorm,
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
        nystrom=False,
        num_landmarks=32,
        kernel_size=None,
        prenorm=True,
        scalenorm=True,
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
            nystrom=nystrom,
            num_landmarks=num_landmarks,
            kernel_size=kernel_size,
            prenorm=prenorm,
            scalenorm=scalenorm,
        )
        self.drop = nn.Dropout(dropout)
        self.predict = nn.Linear(hidden_size, vocab_size)
        reset_parameters(self.named_parameters(), gain=(2.5 * hidden_size) ** -0.5)
        # nn.init.normal_(self.embed.embedding.weight, mean=0, std=hidden_size**-0.5)

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


class TransformerSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        num_layers=6,
        hidden_size=512,
        num_heads=8,
        max_length=512,
        inner_size=2048,
        dropout=0.1,
        nystrom=False,
        num_landmarks=32,
        kernel_size=None,
        prenorm=True,
        scalenorm=True,
        feature_normalization=False,
    ):
        super(TransformerSequenceEncoder, self).__init__()
        self.embed = nn.Linear(input_size, hidden_size)
        self.pe = PositionalEncoding(embedding_dim=hidden_size, max_len=max_length)
        self.feature_norm = None

        if feature_normalization:
            self.feature_norm = ScaleNorm(hidden_size)
        self.transformer_block = Encoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            inner_size=inner_size,
            dropout=dropout,
            nystrom=nystrom,
            num_landmarks=num_landmarks,
            kernel_size=kernel_size,
            prenorm=prenorm,
            scalenorm=scalenorm,
        )
        self.out_size = hidden_size
        reset_parameters(self.named_parameters(), gain=(2.5 * hidden_size) ** -0.5)

    def forward(self, x, attention_mask=None):
        if self.feature_norm:
            x = self.feature_norm(x)

        x = self.embed(x)
        x = self.pe(x)
        out = self.transformer_block(x, attention_mask=attention_mask).mean(dim=1)

        return out


class TransformerTokenSequenceEncoder(nn.Module):
    def __init__(
        self,
        vocab_size=30000,
        max_length=256,
        num_layers=6,
        hidden_size=512,
        num_heads=8,
        inner_size=2048,
        dropout=0.1,
        nystrom=False,
        num_landmarks=32,
        kernel_size=None,
        prenorm=True,
        scalenorm=True,
    ):
        super(TransformerTokenSequenceEncoder, self).__init__()
        self.embed = Embed(
            vocab_size,
            hidden_size,
            scale=math.sqrt(hidden_size),
            dropout=dropout,
            trainable=True,
        )
        self.pe = PositionalEncoding(embedding_dim=hidden_size, max_len=max_length)
        self.transformer_block = Encoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            inner_size=inner_size,
            dropout=dropout,
            nystrom=nystrom,
            num_landmarks=num_landmarks,
            kernel_size=kernel_size,
            prenorm=prenorm,
            scalenorm=scalenorm,
        )
        self.out_size = hidden_size
        reset_parameters(self.named_parameters(), gain=(2.5 * hidden_size) ** -0.5)
        # nn.init.normal_(self.embed.embedding.weight, mean=0, std=hidden_size**-0.5)

    def forward(self, x, attention_mask=None):
        x = self.embed(x)
        x = self.pe(x)
        out = self.transformer_block(x, attention_mask=attention_mask).mean(dim=1)

        return out
