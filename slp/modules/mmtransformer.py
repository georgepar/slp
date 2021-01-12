import math
import torch
import torch.nn as nn

from slp.modules.attention import MultiheadCoAttention
from slp.modules.embed import PositionalEncoding, Embed
from slp.modules.feedforward import PositionwiseFF
from slp.modules.norm import LayerNorm
from slp.modules.util import repeat_layer
from slp.modules.rnn import AttentiveRNN


class Sublayer1(nn.Module):
    def __init__(
        self, hidden_size=512, cross_size=1024, num_heads=8, dropout=0.1, prenorm=True
    ):
        super(Sublayer1, self).__init__()
        self.prenorm = prenorm

        if self.prenorm:
            self.lnorm1 = LayerNorm(hidden_size)
            self.lnorm2 = LayerNorm(cross_size)
        else:
            self.lnorm = LayerNorm(hidden_size)
        self.sublayer = MultiheadCoAttention(
            attention_size=hidden_size,
            input_size=cross_size,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, x, y, attention_mask=None):
        if self.prenorm:
            out = x + self.sublayer(
                self.lnorm1(x), self.lnorm2(y), attention_mask=attention_mask
            )
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
        feedback=False,
        device="cpu",
    ):
        super(MMTransformer3Way, self).__init__()
        self.feedback = feedback

        self.audio_encoder = nn.ModuleList(
            repeat_layer(
                EncoderLayer(
                    hidden_size=hidden_size,
                    cross_size=2 * hidden_size,
                    num_heads=num_heads,
                    inner_size=inner_size,
                    dropout=dropout,
                ),
                num_layers,
            )
        )

        self.text_encoder = nn.ModuleList(
            repeat_layer(
                EncoderLayer(
                    hidden_size=hidden_size,
                    cross_size=2 * hidden_size,
                    num_heads=num_heads,
                    inner_size=inner_size,
                    dropout=dropout,
                ),
                num_layers,
            )
        )

        self.visual_encoder = nn.ModuleList(
            repeat_layer(
                EncoderLayer(
                    hidden_size=hidden_size,
                    cross_size=2 * hidden_size,
                    num_heads=num_heads,
                    inner_size=inner_size,
                    dropout=dropout,
                ),
                num_layers,
            )
        )

    def forward(self, text, audio, visual, attention_mask=None):
        for text_layer, audio_layer, visual_layer in zip(
            self.text_encoder, self.audio_encoder, self.visual_encoder
        ):
            if self.feedback:
                text2 = text_layer(
                    text, torch.cat([audio, visual], dim=-1), attention_mask=attention_mask
                )
                audio2 = audio_layer(
                    audio, torch.cat([text, visual], dim=-1), attention_mask=attention_mask
                )
                visual2 = visual_layer(
                    visual, torch.cat([text, audio], dim=-1), attention_mask=attention_mask
                )

                mt = torch.sigmoid(text2)
                ma = torch.sigmoid(audio2)
                mv = torch.sigmoid(visual2)

                text = text * (ma + mv)
                audio = audio * (mv + mt)
                visual = visual * (ma + mt)

            text1 = text_layer(
                text, torch.cat([audio, visual], dim=-1), attention_mask=attention_mask
            )
            audio1 = audio_layer(
                audio, torch.cat([text, visual], dim=-1), attention_mask=attention_mask
            )
            visual1 = visual_layer(
                visual, torch.cat([text, audio], dim=-1), attention_mask=attention_mask
            )

            text = text1
            audio = audio1
            visual = visual1

        return text, audio, visual

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)



class MMTransformerRnn3Way(nn.Module):
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
        feedback=False,
        device="cpu",
    ):
        super(MMTransformerRnn3Way, self).__init__()
        self.feedback = feedback
        self.device = device

        self.audio_encoder = nn.ModuleList(
            repeat_layer(
                EncoderLayer(
                    hidden_size=hidden_size,
                    cross_size=2 * hidden_size,
                    num_heads=num_heads,
                    inner_size=inner_size,
                    dropout=dropout,
                ),
                num_layers,
            )
        )

        self.text_encoder = nn.ModuleList(
            repeat_layer(
                EncoderLayer(
                    hidden_size=hidden_size,
                    cross_size=2 * hidden_size,
                    num_heads=num_heads,
                    inner_size=inner_size,
                    dropout=dropout,
                ),
                num_layers,
            )
        )

        self.visual_encoder = nn.ModuleList(
            repeat_layer(
                EncoderLayer(
                    hidden_size=hidden_size,
                    cross_size=2 * hidden_size,
                    num_heads=num_heads,
                    inner_size=inner_size,
                    dropout=dropout,
                ),
                num_layers,
            )
        )

        self.output = AttentiveRNN(
            3 * hidden_size,
            3 * hidden_size,
            bidirectional=True,
            dropout=dropout,
            attention=True,
            device=device
        )
        self.out_size = self.output.out_size

    def forward(self, text, audio, visual, lengths, attention_mask=None):
        for text_layer, audio_layer, visual_layer in zip(
            self.text_encoder, self.audio_encoder, self.visual_encoder
        ):
            if self.feedback:
                text2 = text_layer(
                    text, torch.cat([audio, visual], dim=-1), attention_mask=attention_mask
                )
                audio2 = audio_layer(
                    audio, torch.cat([text, visual], dim=-1), attention_mask=attention_mask
                )
                visual2 = visual_layer(
                    visual, torch.cat([text, audio], dim=-1), attention_mask=attention_mask
                )

                mt = torch.sigmoid(text2)
                ma = torch.sigmoid(audio2)
                mv = torch.sigmoid(visual2)

                text = text * (ma + mv)
                audio = audio * (mv + mt)
                visual = visual * (ma + mt)

            text1 = text_layer(
                text, torch.cat([audio, visual], dim=-1), attention_mask=attention_mask
            )
            audio1 = audio_layer(
                audio, torch.cat([text, visual], dim=-1), attention_mask=attention_mask
            )
            visual1 = visual_layer(
                visual, torch.cat([text, audio], dim=-1), attention_mask=attention_mask
            )

            text = text1
            audio = audio1
            visual = visual1

        dat = torch.cat([text, audio, visual], dim=-1)
        out = self.output(dat, lengths)
        return out



    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
