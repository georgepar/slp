import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from slp.modules.embed import Embed
from slp.modules.multimodal import AttentionFuser, AttentionMaskedFuser
from slp.util.pytorch import PackSequence, PadPackedSequence, pad_mask


class RNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        batch_first=True,
        layers=1,
        bidirectional=False,
        merge_bi="cat",
        dropout=0,
        rnn_type="lstm",
        packed_sequence=True,
    ):

        super(RNN, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.merge_bi = merge_bi
        self.rnn_type = rnn_type.lower()

        self.out_size = hidden_size

        if bidirectional and merge_bi == "cat":
            self.out_size = 2 * hidden_size

        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size,
            hidden_size,
            batch_first=batch_first,
            num_layers=layers,
            bidirectional=bidirectional,
        )
        self.drop = nn.Dropout(dropout)
        self.packed_sequence = packed_sequence

        if packed_sequence:
            self.pack = PackSequence(batch_first=batch_first)
            self.unpack = PadPackedSequence(batch_first=batch_first)

    def _merge_bi(self, forward, backward):
        if self.merge_bi == "sum":
            return forward + backward

        return torch.cat((forward, backward), dim=-1)

    def _select_last_unpadded(self, out, lengths):
        gather_dim = 1 if self.batch_first else 0
        gather_idx = (
            (lengths - 1)  # -1 to convert to indices
            .unsqueeze(1)  # (B) -> (B, 1)
            .expand((-1, self.hidden_size))  # (B, 1) -> (B, H)
            # (B, 1, H) if batch_first else (1, B, H)
            .unsqueeze(gather_dim)
        )
        # Last forward for real length or seq (unpadded tokens)
        last_out = out.gather(gather_dim, gather_idx).squeeze(gather_dim)

        return last_out

    def _final_output(self, out, lengths):
        # Collect last hidden state
        # Code adapted from https://stackoverflow.com/a/50950188

        if not self.bidirectional:
            return self._select_last_unpadded(out, lengths)

        forward, backward = (out[..., : self.hidden_size], out[..., self.hidden_size :])
        # Last backward corresponds to first token
        last_backward_out = backward[:, 0, :] if self.batch_first else backward[0, ...]
        # Last forward for real length or seq (unpadded tokens)
        last_forward_out = self._select_last_unpadded(forward, lengths)

        return self._merge_bi(last_forward_out, last_backward_out)

    def merge_hidden_bi(self, out):
        if not self.bidirectional:
            return out

        forward, backward = (out[..., : self.hidden_size], out[..., self.hidden_size :])

        return self._merge_bi(forward, backward)

    def forward(self, x, lengths):
        device = x.device
        self.rnn.flatten_parameters()

        if self.packed_sequence:
            lengths = lengths.to("cpu")
            x, lengths = self.pack(x, lengths)
            lengths = lengths.to(device)
        out, hidden = self.rnn(x)

        if self.packed_sequence:
            out = self.unpack(out, lengths)

        out = self.drop(out)
        last_timestep = self._final_output(out, lengths)
        out = self.merge_hidden_bi(out)

        return out, last_timestep, hidden


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
        p_mmdrop=0.0,
        p_drop_modalities=None,
        multi_modal_drop="mmdrop_hard",
        mmdrop_before_fuse=True,
        mmdrop_after_fuse=False,
        return_hidden=False,
        masking=False,
        m3_sequential=False,
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
            masking=masking,
            m3_sequential=m3_sequential,
        )
        self.rnn = AttentiveRNN(
            self.att_fuser.out_size,
            proj_sz,
            dropout=p_dropout,
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


class AttRnnMaskedFuser(nn.Module):
    def __init__(
        self,
        proj_sz=None,
        p_dropout=0.1,
        p_mmdrop=0.0,
        p_drop_modalities=None,
        multi_modal_drop="mmdrop_hard",
        mmdrop_before_fuse=True,
        mmdrop_after_fuse=False,
        return_hidden=False,
    ):
        super(AttRnnMaskedFuser, self).__init__()
        self.att_fuser = AttentionMaskedFuser(
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
            dropout=p_dropout,
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
        p_mmdrop=0.0,
        p_drop_modalities=None,
        multi_modal_drop="mmdrop_hard",
        mmdrop_before_fuse=True,
        mmdrop_after_fuse=False,
        masking=False,
        m3_sequential=False,
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
            feature_sizes["visual"],
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
            # p_dropout=0.1,
            p_dropout=dropout,
            p_mmdrop=p_mmdrop,
            p_drop_modalities=p_drop_modalities,
            multi_modal_drop=multi_modal_drop,
            mmdrop_before_fuse=mmdrop_before_fuse,
            mmdrop_after_fuse=mmdrop_after_fuse,
            return_hidden=False,
            masking=masking,
            m3_sequential=m3_sequential,
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


class AudioVisualTextMaskedEncoder(nn.Module):
    def __init__(
        self,
        feature_sizes,
        num_layers=1,
        bidirectional=True,
        rnn_type="lstm",
        attention=True,
        hidden_size=100,
        dropout=0.1,
        p_mmdrop=0.0,
        p_drop_modalities=None,
        multi_modal_drop="mmdrop_hard",
        mmdrop_before_fuse=True,
        mmdrop_after_fuse=False,
    ):
        super(AudioVisualTextMaskedEncoder, self).__init__()

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
            feature_sizes["visual"],
            num_layers=num_layers,
            bidirectional=bidirectional,
            merge_bi="sum",
            rnn_type=rnn_type,
            attention=attention,
            hidden_size=hidden_size,
            dropout=dropout,
        )

        self.fuser = AttRnnMaskedFuser(
            proj_sz=hidden_size,
            # p_dropout=0.1,
            p_dropout=dropout,
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
        p_mmdrop=0.0,
        p_drop_modalities=None,
        multi_modal_drop="mmdrop_hard",
        mmdrop_before_fuse=True,
        mmdrop_after_fuse=False,
        masking=False,
        m3_sequential=False,
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
            p_mmdrop=p_mmdrop,
            p_drop_modalities=p_drop_modalities,
            multi_modal_drop=multi_modal_drop,
            mmdrop_before_fuse=mmdrop_before_fuse,
            mmdrop_after_fuse=mmdrop_after_fuse,
            masking=masking,
            m3_sequential=m3_sequential,
        )

        self.classifier = nn.Linear(self.encoder.out_size, num_classes)

    def forward(self, inputs, lengths):
        out = self.encoder(
            inputs["text"], inputs["audio"], inputs["visual"], lengths["text"]
        )

        return self.classifier(out)


class AudioVisualTextMaskedClassifier(nn.Module):
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
        p_mmdrop=0.0,
        p_drop_modalities=None,
        multi_modal_drop="mmdrop_hard",
        mmdrop_before_fuse=True,
        mmdrop_after_fuse=False,
        **kwargs,
    ):
        super(AudioVisualTextMaskedClassifier, self).__init__()

        self.encoder = AudioVisualTextMaskedEncoder(
            feature_sizes,
            num_layers=num_layers,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
            attention=attention,
            hidden_size=hidden_size,
            dropout=dropout,
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
