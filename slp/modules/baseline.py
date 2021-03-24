import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from slp.modules.multimodal import AttentionFuser
from slp.modules.rnn import RNN
from slp.util.pytorch import pad_mask


class Attention(nn.Module):
    """Some Information about Attention"""

    def __init__(
        self,
        attention_size=512,
        input_size=None,
        query_size=None,
        dropout=0.1,
    ):
        super(Attention, self).__init__()

        if input_size is None:
            input_size = attention_size
        if query_size is None:
            query_size = input_size

        self.dk = input_size
        self.k = nn.Linear(input_size, attention_size, bias=False)
        self.q = nn.Linear(query_size, attention_size, bias=False)
        self.v = nn.Linear(input_size, attention_size, bias=False)
        self.drop = nn.Dropout(dropout)
        self._reset_parameters()

    def forward(self, x, queries=None, values=None, attention_mask=None):
        """
        x : (B, L, D)
        queries : (B, L, D)
        values : (B, L, D)
        """
        if queries is None:
            queries = x

        if values is None:
            values = x
        k = self.k(x)  # (B, L, A)
        q = self.q(queries)  # (B, L, A)
        v = self.v(values)  # (B, L, A)

        # weights => (B, L, L)

        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.dk)

        if attention_mask is not None:
            scores = scores + ((1 - attention_mask.unsqueeze(1)) * -1e5)
        scores = F.softmax(scores, dim=-1)
        scores = self.drop(scores)

        # out => (B, L, A)
        out = torch.bmm(scores, v)

        return out, scores

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.v.weight)


class AttentiveRNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        batch_first=True,
        layers=1,
        bidirectional=False,
        merge_bi="cat",
        dropout=0.1,
        rnn_type="lstm",
        packed_sequence=True,
        attention=False,
        return_hidden=False,
        **extra_args,
    ):
        super(AttentiveRNN, self).__init__()
        self.rnn = RNN(
            input_size,
            hidden_size,
            batch_first=batch_first,
            layers=layers,
            merge_bi=merge_bi,
            bidirectional=bidirectional,
            dropout=dropout,
            rnn_type=rnn_type,
            packed_sequence=packed_sequence,
        )
        self.out_size = self.rnn.out_size
        self.attention = None
        self.return_hidden = return_hidden

        if attention:
            self.attention = Attention(attention_size=self.out_size, dropout=dropout)

    def forward(self, x, lengths):
        out, last_hidden, _ = self.rnn(x, lengths)

        if self.attention is not None:
            out, _ = self.attention(out, attention_mask=pad_mask(lengths))

            if not self.return_hidden:
                out = out.sum(1)
        else:
            if not self.return_hidden:
                out = last_hidden

        return out


class AttRnnFuser(nn.Module):
    def __init__(
        self,
        proj_sz=None,
        p_dropout=0.1,
        p_mmdrop=0,
        p_drop_modalities=None,
        multi_modal_drop="mmdrop_hard",
        mmdrop_before_fuse=True,
        mmdrop_after_fuse=False,
        return_hidden=False,
    ):
        super(AttRnnFuser, self).__init__()
        self.att_fuser = AttentionFuser(
            proj_sz=proj_sz,
            residual=1,
            return_hidden=True,
            p_dropout=p_dropout,
            p_mmdrop=p_mmdrop,
            p_drop_modalities=p_drop_modalities,
            multi_modal_drop=multi_modal_drop,
            mmdrop_before_fuse=mmdrop_before_fuse,
            mmdrop_after_fuse=mmdrop_after_fuse,
        )
        self.rnn = AttentiveRNN(
            self.att_fuser.out_size,
            proj_sz,
            bidirectional=True,
            merge_bi="cat",
            attention=True,
            return_hidden=return_hidden,
        )
        self.out_size = self.rnn.out_size

    def forward(self, txt, au, vi, lengths):
        att = self.att_fuser(txt, au, vi)  # B x L x 7 * D
        out = self.rnn(att, lengths)  # B x L x 2 * D

        return out


class AudioEncoder(nn.Module):
    def __init__(
        self,
        input_size=74,
        num_layers=1,
        bidirectional=True,
        merge_bi="sum",
        rnn_type="lstm",
        attention=True,
        hidden_size=100,
        dropout=0.1,
    ):
        super(AudioEncoder, self).__init__()
        self.audio = AttentiveRNN(
            input_size,
            hidden_size,
            batch_first=True,
            layers=num_layers,
            merge_bi=merge_bi,
            bidirectional=bidirectional,
            dropout=dropout,
            rnn_type=rnn_type,
            packed_sequence=True,
            attention=attention,
            return_hidden=True,
        )
        self.out_size = self.audio.out_size

    def forward(self, x, lengths):
        x = self.audio(x, lengths)

        return x


class VisualEncoder(nn.Module):
    def __init__(
        self,
        input_size=35,
        num_layers=1,
        bidirectional=True,
        merge_bi="sum",
        rnn_type="lstm",
        attention=True,
        hidden_size=100,
        dropout=0.1,
    ):
        super(VisualEncoder, self).__init__()
        self.visual = AttentiveRNN(
            input_size,
            hidden_size,
            batch_first=True,
            layers=num_layers,
            merge_bi=merge_bi,
            bidirectional=bidirectional,
            dropout=dropout,
            rnn_type=rnn_type,
            packed_sequence=True,
            attention=attention,
            return_hidden=True,
        )
        self.out_size = self.visual.out_size

    def forward(self, x, lengths):
        x = self.visual(x, lengths)

        return x


class GloveEncoder(nn.Module):
    def __init__(
        self,
        input_size=300,
        num_layers=1,
        bidirectional=True,
        merge_bi="sum",
        rnn_type="lstm",
        attention=True,
        hidden_size=100,
        dropout=0.1,
    ):
        super(GloveEncoder, self).__init__()
        self.text = AttentiveRNN(
            input_size,
            hidden_size,
            batch_first=True,
            layers=num_layers,
            merge_bi=merge_bi,
            bidirectional=bidirectional,
            dropout=dropout,
            rnn_type=rnn_type,
            packed_sequence=True,
            attention=attention,
            return_hidden=True,
        )
        self.out_size = self.text.out_size

    def forward(self, x, lengths):
        x = self.text(x, lengths)

        return x


class AudioVisualTextEncoder(nn.Module):
    def __init__(
        self,
        feature_sizes,
        num_layers=1,
        bidirectional=True,
        rnn_type="lstm",
        attention=True,
        hidden_size=100,
        dropout=0.1,
        p_dropout=0.1,
        p_mmdrop=0,
        p_drop_modalities=None,
        multi_modal_drop="mmdrop_hard",
        mmdrop_before_fuse=True,
        mmdrop_after_fuse=False,
    ):
        super(AudioVisualTextEncoder, self).__init__()

        self.text = GloveEncoder(
            feature_sizes["text"],
            num_layers=num_layers,
            bidirectional=bidirectional,
            merge_bi="sum",
            rnn_type=rnn_type,
            attention=attention,
            hidden_size=hidden_size,
            dropout=dropout,
        )

        self.audio = AudioEncoder(
            feature_sizes["audio"],
            num_layers=num_layers,
            bidirectional=bidirectional,
            merge_bi="sum",
            rnn_type=rnn_type,
            attention=attention,
            hidden_size=hidden_size,
            dropout=dropout,
        )

        self.visual = VisualEncoder(
            feature_sizes["audio"],
            num_layers=num_layers,
            bidirectional=bidirectional,
            merge_bi="sum",
            rnn_type=rnn_type,
            attention=attention,
            hidden_size=hidden_size,
            dropout=dropout,
        )

        self.fuser = AttRnnFuser(
            proj_sz=hidden_size,
            p_dropout=p_dropout,
            p_mmdrop=p_mmdrop,
            p_drop_modalities=p_drop_modalities,
            multi_modal_drop=multi_modal_drop,
            mmdrop_before_fuse=mmdrop_before_fuse,
            mmdrop_after_fuse=mmdrop_after_fuse,
            return_hidden=False,
        )

        self.out_size = self.fuser.out_size

    def _encode(self, txt, au, vi, lengths):
        txt = self.text(txt, lengths)
        au = self.audio(au, lengths)
        vi = self.visual(vi, lengths)

        return txt, au, vi

    def _fuse(self, txt, au, vi, lengths):
        fused = self.fuser(txt, au, vi, lengths)

        return fused

    def forward(self, txt, au, vi, lengths):
        txt, au, vi = self._encode(txt, au, vi, lengths)
        fused = self._fuse(txt, au, vi, lengths)

        return fused


class AudioVisualTextClassifier(nn.Module):
    def __init__(
        self,
        feature_sizes,
        num_classes,
        num_layers=1,
        bidirectional=True,
        rnn_type="lstm",
        attention=True,
        hidden_size=100,
        dropout=0.1,
        p_dropout=0.1,
        p_mmdrop=0,
        p_drop_modalities=None,
        multi_modal_drop="mmdrop_hard",
        mmdrop_before_fuse=True,
        mmdrop_after_fuse=False,
        **kwargs,
    ):
        super(AudioVisualTextClassifier, self).__init__()

        self.encoder = AudioVisualTextEncoder(
            feature_sizes,
            num_layers=num_layers,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
            attention=attention,
            hidden_size=hidden_size,
            dropout=dropout,
            p_dropout=p_dropout,
            p_mmdrop=p_mmdrop,
            p_drop_modalities=p_drop_modalities,
            multi_modal_drop=multi_modal_drop,
            mmdrop_before_fuse=mmdrop_before_fuse,
            mmdrop_after_fuse=mmdrop_after_fuse,
        )

        self.classifier = nn.Linear(self.encoder.out_size, num_classes)

    def forward(self, inputs, lengths):
        out = self.encoder(
            inputs["text"], inputs["audio"], inputs["visual"], lengths["text"]
        )

        return self.classifier(out)
