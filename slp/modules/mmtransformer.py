import math
import torch
import torch.nn as nn

from slp.modules.attention import MultiheadCoAttention
from slp.modules.embed import PositionalEncoding, Embed
from slp.modules.feedforward import PositionwiseFF
from slp.modules.norm import LayerNorm
from slp.modules.util import repeat_layer


class Sublayer1(nn.Module):
    def __init__(self, hidden_size=512, cross_size=1024, num_heads=8, dropout=0.1, prenorm=True):
        super(Sublayer1, self).__init__()
        self.prenorm = prenorm
        if self.prenorm:
            self.lnorm1 = LayerNorm(hidden_size)
            self.lnorm2 = LayerNorm(cross_size)
        else:
            self.lnorm = LayerNorm(hidden_size)
        self.sublayer = MultiheadCoAttention(
            attention_size=hidden_size,
            num_heads=num_heads,
            query_size=cross_size,
            dropout=dropout,
        )

    def forward(self, x, y, attention_mask=None):
        if self.prenorm:
            out = x + self.sublayer(self.lnorm1(x), self.lnorm2(y), attention_mask=attention_mask)
        else:
            out = self.lnorm(x + self.sublayer(x, y, attention_mask=attention_mask))
        return out

class Sublayer2(nn.Module):
    def __init__(self, hidden_size=512, inner_size=2048, dropout=0.1, prenorm=True):
        super(Sublayer2, self).__init__()
        self.lnorm = LayerNorm(hidden_size)
        self.prenorm = prenorm
        self.sublayer = PositionwiseFF(hidden_size, inner_size, dropout=dropout)

    def forward(self, x):
        if self.prenorm:
            out = x + self.sublayer(self.lnorm(x))
        else:
            out = self.lnorm(x + self.sublayer(x))
        return out

class EncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        cross_size=1024,
        num_heads=8,
        inner_size=2048,
        dropout=0.1,
    ):
        super(EncoderLayer, self).__init__()
        self.l1 = Sublayer1(
            hidden_size=hidden_size,
            cross_size=cross_size,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.l2 = Sublayer2(
            hidden_size=hidden_size, inner_size=inner_size, dropout=dropout
        )

    def forward(self, x, y, attention_mask=None):
        out = self.l1(x, y, attention_mask=attention_mask)
        out = self.l2(out)

        return out


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers=6,
        hidden_size=512,
        cross_size=1024,
        num_heads=8,
        inner_size=2048,
        dropout=0.1,
    ):
        super(Encoder, self).__init__()
        self.encoder = nn.ModuleList(
            repeat_layer(
                EncoderLayer(
                    hidden_size=hidden_size,
                    cross_size=cross_size,
                    num_heads=num_heads,
                    inner_size=inner_size,
                    dropout=dropout,
                ),
                num_layers,
            )
        )

    def forward(self, x, y, attention_mask=None):
        for i, layer in enumerate(self.encoder):
            x = layer(x, y, attention_mask=attention_mask)
        return x


class MMEncoder(nn.Module):
    def __init__(
        self,
        num_layers=6,
        hidden_size=512,
        cross_size=1024,
        num_heads=8,
        inner_size=2048,
        dropout=0.1,
        device="cpu",
    ):
        super(MMEncoder, self).__init__()
        self.encoder_block = Encoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            cross_size=cross_size,
            num_heads=num_heads,
            inner_size=inner_size,
            dropout=dropout,
        )
        self.drop = nn.Dropout(dropout)
        self._reset_parameters()

    def forward(self, x, y, attention_mask=None):
        # x -> main modality
        # y -> cross modality
        # both x, y embedded and passed through pos embeddings
        out = self.encoder_block(x, y, attention_mask=attention_mask)
        out = self.drop(out)

        return out

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class MMTransformer3Way(nn.Module):
    def __init__(
        self,
        text_size,
        audio_size,
        visual_size,
        hidden_size=512,
        max_length=256,
        num_layers=6,
        num_heads=8,
        inner_size=2048,
        dropout=0.1,
        device="cpu",
    ):
        super(MMTransformer3Way, self).__init__()

        self.pe = PositionalEncoding(
            max_length, embedding_dim=hidden_size, device=device
        )

        self.audio_embed = nn.Linear(audio_size, hidden_size)
        #self.text_embed = nn.Linear(text_size, hidden_size)
        self.visual_embed = nn.Linear(visual_size, hidden_size)

        self.audio_encoder = MMEncoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            cross_size=2 * hidden_size,
            num_heads=num_heads,
            inner_size=inner_size,
            dropout=dropout,
            device=device,
        )

        self.text_encoder = MMEncoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            cross_size=2 * hidden_size,
            num_heads=num_heads,
            inner_size=inner_size,
            dropout=dropout,
            device=device,
        )

        self.visual_encoder = MMEncoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            cross_size=2 * hidden_size,
            num_heads=num_heads,
            inner_size=inner_size,
            dropout=dropout,
            device=device,
        )

    def forward(self, text, audio, visual, attention_mask=None):
        # text = self.text_embed(text)
        audio = self.audio_embed(audio)
        visual = self.visual_embed(visual)

        text = self.pe(text)
        audio = self.pe(audio)
        visual = self.pe(visual)

        text = self.text_encoder(
            text, torch.cat([audio, visual], dim=-1), attention_mask=attention_mask
        )

        audio = self.audio_encoder(
            audio, torch.cat([text, visual], dim=-1), attention_mask=attention_mask
        )

        visual = self.visual_encoder(
            audio, torch.cat([text, audio], dim=-1), attention_mask=attention_mask
        )

        return text, audio, visual
