import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from slp.modules.attention import Attention, SymmetricAttention
from slp.modules.rnn import RNN, AttentiveRNN


def make_h0(seq, lstm=True):
    h0 = seq.sum(1).unsqueeze(0).repeat(2, 1, 1)

    if lstm:
        c0 = torch.clone(h0)

        return (h0, c0)

    return h0


class MultimodalDropout(nn.Module):
    def __init__(self, p=0.5, n_modalities=3, device="cpu"):
        super(MultimodalDropout, self).__init__()
        self.p = p
        self.device = device
        self.n_modalities = n_modalities

    def forward(self, *mods):
        mods = list(mods)

        if self.training:
            if random.random() < self.p:
                for i in range(mods[0].size(0)):
                    m = random.randint(0, self.n_modalities - 1)
                    mask = torch.ones_like(mods[m])
                    mask[i] = 0.0
                    mods[m] = mods[m] * mask

        return mods


class FeedbackUnit(nn.Module):
    def __init__(
        self,
        hidden_dim,
        mod1_sz,
        use_self=False,
        mask_type="sigmoid",
        dropout=0.1,
        device="cpu",
    ):
        super(FeedbackUnit, self).__init__()
        self.use_self = use_self
        self.mask_type = mask_type
        self.mod1_sz = mod1_sz
        self.hidden_dim = hidden_dim

        if mask_type == "rnn" or mask_type == "sum_rnn":
            self.mask1 = RNN(hidden_dim, mod1_sz, dropout=dropout, device=device)
            self.mask2 = RNN(hidden_dim, mod1_sz, dropout=dropout, device=device)

            if use_self:
                self.mask_self = RNN(
                    hidden_dim, mod1_sz, dropout=dropout, device=device
                )
        else:
            self.mask1 = nn.Linear(hidden_dim, mod1_sz)
            self.mask2 = nn.Linear(hidden_dim, mod1_sz)

            if use_self:
                self.mask_self = nn.Linear(hidden_dim, mod1_sz)

        mask_fn = {
            "sum_sigmoid": self._sum_sigmoid_mask,
            "sum_rnn": self._sum_rnn_mask,
        }

        self.get_mask = mask_fn[self.mask_type]

    def _sum_rnn_mask(self, x, y, z, x_high=None, lengths=None):
        oy, _, _ = self.mask1(y, lengths)
        oz, _, _ = self.mask2(z, lengths)

        lg = (torch.sigmoid(oy) + torch.sigmoid(oz)) * 0.5

        if self.use_self:
            ox, _, _ = self.mask_self(x_high, lengths)
            lg = lg + torch.sigmoid(ox)

        mask = lg

        return mask

    def _sum_sigmoid_mask(self, x, y, z, x_high=None, lengths=None):
        y = self.mask1(y)
        z = self.mask2(z)
        mask1 = torch.sigmoid(y)
        mask2 = torch.sigmoid(z)
        mask = (mask1 + mask2) * 0.5

        if self.use_self:
            m = torch.sigmoid(self.mask_self(x_high))
            mask = mask + m

        return mask

    def forward(self, x, y, z, x_high=None, lengths=None):
        mask = self.get_mask(x, y, z, x_high=x_high, lengths=lengths)

        if self.mask_type == "attention":
            x = torch.bmm(mask, x)
        else:
            x = x * mask

        return x


class Feedback(nn.Module):
    def __init__(
        self,
        hidden_dim,
        mod1_sz,
        mod2_sz,
        mod3_sz,
        use_self=False,
        mask_type="sigmoid",
        dropout=0.1,
        device="cpu",
    ):
        super(Feedback, self).__init__()
        self.f1 = FeedbackUnit(
            hidden_dim,
            mod1_sz,
            use_self=use_self,
            mask_type=mask_type,
            dropout=dropout,
            device=device,
        )
        self.f2 = FeedbackUnit(
            hidden_dim,
            mod2_sz,
            use_self=use_self,
            mask_type=mask_type,
            dropout=dropout,
            device=device,
        )
        self.f3 = FeedbackUnit(
            hidden_dim,
            mod3_sz,
            use_self=use_self,
            mask_type=mask_type,
            dropout=dropout,
            device=device,
        )

    def forward(self, low_x, low_y, low_z, hi_x, hi_y, hi_z, lengths=None):
        x = self.f1(low_x, hi_y, hi_z, x_high=hi_x, lengths=lengths)
        y = self.f2(low_y, hi_x, hi_z, x_high=hi_y, lengths=lengths)
        z = self.f3(low_z, hi_x, hi_y, x_high=hi_z, lengths=lengths)

        return x, y, z


class Conv1dProj(nn.Module):
    def __init__(self, mod1_sz, mod2_sz, mod3_sz, proj_sz):
        super(Conv1dProj, self).__init__()
        self.p1 = nn.Conv1d(mod1_sz, proj_sz, kernel_size=1, padding=0, bias=False)
        self.p2 = nn.Conv1d(mod2_sz, proj_sz, kernel_size=1, padding=0, bias=False)
        self.p3 = nn.Conv1d(mod3_sz, proj_sz, kernel_size=1, padding=0, bias=False)

    def forward(self, x, y, z):
        x = self.p1(x.transpose(1, 2)).transpose(1, 2)
        y = self.p2(y.transpose(1, 2)).transpose(1, 2)
        z = self.p3(z.transpose(1, 2)).transpose(1, 2)

        return x, y, z


class LinearProj(nn.Module):
    def __init__(self, mod1_sz, mod2_sz, mod3_sz, proj_sz):
        super(LinearProj, self).__init__()
        self.p1 = nn.Linear(mod1_sz, proj_sz)
        self.p2 = nn.Linear(mod2_sz, proj_sz)
        self.p3 = nn.Linear(mod3_sz, proj_sz)

    def forward(self, x, y, z):
        x = self.p1(x)
        y = self.p2(y)
        z = self.p3(z)

        return x, y, z


class ModalityWeights(nn.Module):
    def __init__(
        self,
        mod1_sz,
        mod2_sz,
        mod3_sz,
        proj_sz=None,
        modality_weights=False,
    ):
        super(ModalityWeights, self).__init__()
        self.proj, self.mod_w = None, None
        self.proj_sz = mod1_sz if proj_sz is None else proj_sz

        if proj_sz is not None:
            self.proj = LinearProj(mod1_sz, mod2_sz, mod3_sz, self.proj_sz)

        if modality_weights:
            self.mod_w = nn.Linear(self.proj_sz, 1)

    def forward(self, x, y, z):
        if self.proj:
            x, y, z = self.proj(x, y, z)

        if self.mod_w:
            w = self.mod_w(
                torch.cat([x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)], dim=1)
            )
            w = F.softmax(w, dim=1)
            x = x * w[:, 0, ...]
            y = y * w[:, 1, ...]
            z = z * w[:, 2, ...]

        return x, y, z


class CatFuser(nn.Module):
    def __init__(
        self,
        mod1_sz,
        mod2_sz,
        mod3_sz,
        proj_sz=None,
        modality_weights=False,
        device="cpu",
        mmdrop=0,
        extra_args=None,
    ):
        super(CatFuser, self).__init__()
        self.mw = ModalityWeights(
            mod1_sz,
            mod2_sz,
            mod3_sz,
            proj_sz=proj_sz,
            modality_weights=modality_weights,
        )
        self.mmdrop = MultimodalDropout(p=mmdrop, n_modalities=3, device=device)
        self.out_size = mod1_sz + mod2_sz + mod3_sz if proj_sz is None else 3 * proj_sz

    def forward(self, x, y, z):
        x, y, z = self.mmdrop(x, y, z)
        x, y, z = self.mw(x, y, z)

        return torch.cat([x, y, z], dim=1)


class AddFuser(nn.Module):
    def __init__(
        self,
        mod1_sz,
        mod2_sz,
        mod3_sz,
        proj_sz=None,
        modality_weights=False,
        device="cpu",
        mmdrop=0,
        extra_args=None,
    ):
        super(AddFuser, self).__init__()
        self.mw = ModalityWeights(
            mod1_sz,
            mod2_sz,
            mod3_sz,
            proj_sz=proj_sz,
            modality_weights=False,
        )
        self.mmdrop = MultimodalDropout(p=mmdrop, n_modalities=3, device=device)
        self.out_size = mod1_sz if proj_sz is None else proj_sz

    def forward(self, x, y, z):
        x, y, z = self.mmdrop(x, y, z)
        x, y, z = self.mw(x, y, z)

        return x + y + z


class RnnFuser(nn.Module):
    def __init__(
        self,
        sizes,
        proj_sz=None,
        modality_weights=False,
        device="cpu",
        proj_type="linear",
        mmdrop=0,
        extra_args=None,
    ):
        super(RnnFuser, self).__init__()
        input_size = sum(sizes)

        if proj_sz is None:
            proj_sz = input_size
        self.rnn = AttentiveRNN(
            input_size,
            proj_sz,
            bidirectional=True,
            merge_bi="cat",
            attention=True,
            device=device,
        )
        self.mmdrop = MultimodalDropout(
            p=mmdrop, n_modalities=len(sizes), device=device
        )
        self.out_size = self.rnn.out_size

    def forward(self, *inputs):
        lengths = inputs[-1]
        mods = self.mmdrop(*inputs[:-1])
        inp = torch.cat(mods, dim=-1)
        out = self.rnn(inp, lengths)

        return out


class AttentionFuser(nn.Module):
    def __init__(
        self,
        proj_sz=None,
        residual=1,
        return_hidden=True,
        mmdrop=0,
        mmdrop_text_only=False,
        device="cpu",
        return_cross_attentions=False,
    ):
        super(AttentionFuser, self).__init__()
        self.return_hidden = return_hidden
        self.ta = SymmetricAttention(
            attention_size=proj_sz,
            dropout=0.1,
            residual=residual,
            layernorm=False,
        )

        self.va = SymmetricAttention(
            attention_size=proj_sz,
            dropout=0.1,
            residual=residual,
            layernorm=False,
        )

        self.tv = SymmetricAttention(
            attention_size=proj_sz,
            dropout=0.1,
            residual=residual,
            layernorm=False,
        )

        self.tav = Attention(
            attention_size=proj_sz,
            dropout=0.1,
        )

        self.mmdrop_text_only = mmdrop_text_only

        self.n_modalities = 7
        if mmdrop_text_only:
            self.mmdrop = MultimodalDropout(
                p=mmdrop, n_modalities=4, device=device
            )
        else:
            self.mmdrop = MultimodalDropout(
                p=mmdrop, n_modalities=self.n_modalities, device=device
            )

        self.return_cross_attentions = return_cross_attentions
        self.out_size = self.n_modalities * proj_sz

    def forward(self, txt, au, vi):
        # txt, au, vi = self.mmdrop(txt, au, vi)
        ta, at = self.ta(txt, au)
        va, av = self.va(vi, au)
        tv, vt = self.tv(txt, vi)

        av = va + av
        tv = vt + tv
        ta = ta + at

        tav, _ = self.tav(txt, queries=av)

        # Sum weighted attention hidden states

        if not self.return_hidden:
            txt = txt.sum(1)
            au = au.sum(1)
            vi = vi.sum(1)
            ta = ta.sum(1)
            av = av.sum(1)
            tv = tv.sum(1)
            tav = tav.sum(1)

        if self.mmdrop_text_only:
            txt, ta, tv, tav = self.mmdrop(txt, ta, tv, tav)
        else:
            txt, au, vi, ta, tv, av, tav = self.mmdrop(txt, au, vi, ta, tv, av, tav)

        # B x L x 7*D

        fused = torch.cat([txt, au, vi, ta, tv, av, tav], dim=-1)

        if self.return_cross_attentions:
            return fused, [txt, au, vi, ta, tv, av, tav]
        else:
            return fused, None


class AttRnnFuser(nn.Module):
    def __init__(
        self,
        proj_sz=None,
        residual=1,
        mmdrop=0,
        mmdrop_text_only=False,
        device="cpu",
        return_hidden=False,
        return_cross_attentions=False,
        init_tav=False,
    ):
        super(AttRnnFuser, self).__init__()
        self.att_fuser = AttentionFuser(
            proj_sz=proj_sz,
            residual=residual,
            return_hidden=True,
            mmdrop=mmdrop,
            mmdrop_text_only=mmdrop_text_only,
            return_cross_attentions=return_cross_attentions,
            device=device,
        )

        self.init_tav = init_tav
        input_dim = self.att_fuser.out_size

        if self.init_tav:
            sz = self.att_fuser.out_size
            nmod = self.att_fuser.n_modalities
            input_dim = int(sz * (nmod - 1) / nmod)

        self.rnn = AttentiveRNN(
            input_dim,
            proj_sz,
            bidirectional=True,
            merge_bi="cat",
            attention=True,
            device=device,
            return_hidden=return_hidden,
        )
        self.return_cross_attentions = return_cross_attentions
        self.out_size = self.rnn.out_size

    def forward(self, txt, au, vi, lengths):
        _, att = self.att_fuser(txt, au, vi)  # B x L x 7 * D
        h = att
        h0 = None

        if self.return_cross_attentions:
            if self.init_tav:
                h = torch.cat(att[:-1], dim=-1)
                h0 = make_h0(att[-1])
            else:
                h = torch.cat(att, dim=-1)
        out = self.rnn(h, lengths, initial_hidden=h0)  # B x L x 2 * D

        return out, att


class AudioEncoder(nn.Module):
    def __init__(self, cfg, device="cpu"):
        super(AudioEncoder, self).__init__()
        self.audio = AttentiveRNN(
            cfg["input_size"],
            cfg["hidden_size"],
            batch_first=True,
            layers=cfg["layers"],
            merge_bi="sum",
            bidirectional=cfg["bidirectional"],
            dropout=cfg["dropout"],
            rnn_type=cfg["rnn_type"],
            packed_sequence=True,
            attention=cfg["attention"],
            device=device,
            return_hidden=cfg["return_hidden"],
        )
        self.out_size = self.audio.out_size

        self.bn = None

        if cfg["batchnorm"]:
            self.bn = nn.BatchNorm1d(cfg["input_size"])

    def forward(self, x, lengths, initial_hidden=None):
        if self.bn is not None:
            x = self.bn(x.view(-1, x.size(2), x.size(1))).view(-1, x.size(1), x.size(2))
        x = self.audio(x, lengths, initial_hidden=initial_hidden)

        return x


class VisualEncoder(nn.Module):
    def __init__(self, cfg, device="cpu"):
        super(VisualEncoder, self).__init__()
        self.visual = AttentiveRNN(
            cfg["input_size"],
            cfg["hidden_size"],
            batch_first=True,
            layers=cfg["layers"],
            merge_bi="sum",
            bidirectional=cfg["bidirectional"],
            dropout=cfg["dropout"],
            rnn_type=cfg["rnn_type"],
            packed_sequence=True,
            attention=cfg["attention"],
            device=device,
            return_hidden=cfg["return_hidden"],
        )
        self.out_size = self.visual.out_size

        self.bn = None

        if cfg["batchnorm"]:
            self.bn = nn.BatchNorm1d(cfg["input_size"])

    def forward(self, x, lengths, initial_hidden=None):
        if self.bn is not None:
            x = self.bn(x.view(-1, x.size(2), x.size(1))).view(-1, x.size(1), x.size(2))
        x = self.visual(x, lengths, initial_hidden=initial_hidden)

        return x


class GloveEncoder(nn.Module):
    def __init__(self, cfg, device="cpu"):
        super(GloveEncoder, self).__init__()
        self.text = AttentiveRNN(
            cfg["input_size"],
            cfg["hidden_size"],
            batch_first=True,
            layers=cfg["layers"],
            merge_bi="sum",
            bidirectional=cfg["bidirectional"],
            dropout=cfg["dropout"],
            rnn_type=cfg["rnn_type"],
            packed_sequence=True,
            attention=cfg["attention"],
            device=device,
            return_hidden=cfg["return_hidden"],
        )
        self.out_size = self.text.out_size

    def forward(self, x, lengths, initial_hidden=None):
        x = self.text(x, lengths, initial_hidden=initial_hidden)

        return x


class AudioVisualTextEncoder(nn.Module):
    def __init__(
        self,
        embeddings=None,
        vocab_size=None,
        audio_cfg=None,
        text_cfg=None,
        visual_cfg=None,
        fuse_cfg=None,
        feedback=False,
        text_mode="glove",
        device="cpu",
    ):
        super(AudioVisualTextEncoder, self).__init__()
        # For now model dim == text dim (the largest). In the future this can be done
        # with individual projection layers for each modality
        assert (
            text_cfg["attention"] and audio_cfg["attention"] and visual_cfg["attention"]
        ), "Use attention pls."

        self.feedback = feedback
        self.feedback_h0 = False
        self.feedback_mask = True

        if feedback:
            self.feedback_no_grad = fuse_cfg.get("feedback_no_grad", False)
            self.feedback_h0 = fuse_cfg.get("feedback_h0", False)
            self.feedback_mask = fuse_cfg.get("feedback_mask", True)

        self.return_cross_attentions = True
        mmdrop_text_only = fuse_cfg.get("mmdrop_text_only", False)

        if self.feedback_h0:
            assert fuse_cfg["method"] in [
                "attrnn",
                "att",
            ], "Only attrnn, att supported when feedback_h0"

        if not (self.feedback_h0 or self.feedback_mask) and feedback:
            raise ValueError(
                "At least one of fuse/feedback_mask or fuse/feedback_h0"
                " should be enabled when using feedback"
            )

        text_cfg["orig_size"] = text_cfg.get("orig_size", text_cfg["input_size"])
        audio_cfg["orig_size"] = audio_cfg.get("orig_size", audio_cfg["input_size"])
        visual_cfg["orig_size"] = visual_cfg.get("orig_size", visual_cfg["input_size"])

        if fuse_cfg["projection_type"] == "conv":
            self.proj = Conv1dProj(
                text_cfg["orig_size"],
                audio_cfg["orig_size"],
                visual_cfg["orig_size"],
                fuse_cfg["model_dim"],
            )
        elif fuse_cfg["projection_type"] == "linear":
            self.proj = LinearProj(
                text_cfg["orig_size"],
                audio_cfg["orig_size"],
                visual_cfg["orig_size"],
                fuse_cfg["model_dim"],
            )
        else:
            self.proj = None

        if self.proj is not None:
            text_cfg["input_size"] = fuse_cfg["model_dim"]
            audio_cfg["input_size"] = fuse_cfg["model_dim"]
            visual_cfg["input_size"] = fuse_cfg["model_dim"]

        text_cfg["return_hidden"] = True
        audio_cfg["return_hidden"] = True
        visual_cfg["return_hidden"] = True

        if text_mode == "glove":
            self.text = GloveEncoder(text_cfg, device=device)
        else:
            raise ValueError("Only glove supported for now")

        self.audio = AudioEncoder(audio_cfg, device=device)

        self.visual = VisualEncoder(visual_cfg, device=device)

        if fuse_cfg["method"] == "cat":
            self.fuser = CatFuser(
                self.text.out_size,
                self.audio.out_size,
                self.visual.out_size,
                proj_sz=fuse_cfg["projection_size"],
                modality_weights=fuse_cfg["modality_weights"],
                device=device,
                mmdrop=fuse_cfg["mmdrop"],
                extra_args=fuse_cfg,
            )

        elif fuse_cfg["method"] == "add":
            self.fuser = AddFuser(
                self.text.out_size,
                self.audio.out_size,
                self.visual.out_size,
                proj_sz=fuse_cfg["projection_size"],
                modality_weights=fuse_cfg["modality_weights"],
                device=device,
                mmdrop=fuse_cfg["mmdrop"],
                extra_args=fuse_cfg,
            )

        elif fuse_cfg["method"] == "att":
            self.fuser = AttentionFuser(
                proj_sz=fuse_cfg["projection_size"],
                residual=fuse_cfg["residual"],
                mmdrop=fuse_cfg["mmdrop"],
                mmdrop_text_only=mmdrop_text_only,
                return_hidden=False,
                device=device,
                return_cross_attentions=self.return_cross_attentions,
            )
        elif fuse_cfg["method"] == "rnn":
            self.fuser = RnnFuser(
                [self.text.out_size, self.audio.out_size, self.visual.out_size],
                proj_sz=fuse_cfg["projection_size"],
                mmdrop=fuse_cfg["mmdrop"],
                device=device,
            )
        elif fuse_cfg["method"] == "attrnn":
            self.fuser = AttRnnFuser(
                proj_sz=fuse_cfg["projection_size"],
                residual=fuse_cfg["residual"],
                mmdrop=fuse_cfg["mmdrop"],
                mmdrop_text_only=mmdrop_text_only,
                device=device,
                return_cross_attentions=self.return_cross_attentions,
                init_tav=fuse_cfg.get("init_tav", False),
            )
        else:
            raise ValueError('Supported fuse techniques: ["cat", "add"]')

        self.fuse_method = fuse_cfg["method"]

        self.out_size = self.fuser.out_size

        if feedback:
            self.fm = Feedback(
                fuse_cfg["projection_size"],
                text_cfg["orig_size"],
                audio_cfg["orig_size"],
                visual_cfg["orig_size"],
                use_self=fuse_cfg["self_feedback"],
                mask_type=fuse_cfg["feedback_type"],
                dropout=0.1,
                device=device,
            )

    def _encode(self, txt, au, vi, lengths, txt_h0=None, au_h0=None, vi_h0=None):
        if self.proj is not None:
            txt, au, vi = self.proj(txt, au, vi)
        txt = self.text(txt, lengths, initial_hidden=txt_h0)
        au = self.audio(au, lengths, initial_hidden=au_h0)
        vi = self.visual(vi, lengths, initial_hidden=vi_h0)

        return txt, au, vi

    def _fuse(self, txt, au, vi, lengths):
        cross_attentions = None

        if self.fuse_method in ["cat", "sum"]:
            # Sum weighted attention hidden states
            fused = self.fuser(txt.sum(1), au.sum(1), vi.sum(1))
        elif self.fuse_method == "att":
            fused, cross_attentions = self.fuser(txt, au, vi)
        else:
            fused, cross_attentions = self.fuser(txt, au, vi, lengths)

        return fused, cross_attentions

    def forward(self, txt, au, vi, lengths):
        txt_h0, au_h0, vi_h0 = None, None, None

        if self.feedback:
            for _ in range(1):
                if self.feedback_no_grad:
                    with torch.no_grad():
                        txt1, au1, vi1 = self._encode(txt, au, vi, lengths)
                else:
                    txt1, au1, vi1 = self._encode(txt, au, vi, lengths)

                if self.feedback_mask:
                    txt, au, vi = self.fm(txt, au, vi, txt1, au1, vi1, lengths=lengths)

                if self.feedback_h0:
                    if self.feedback_no_grad:
                        with torch.no_grad():
                            _, [_, _, _, ta, tv, av, _] = self._fuse(
                                txt1, au1, vi1, lengths
                            )
                    else:
                        _, [_, _, _, ta, tv, av, _] = self._fuse(
                            txt1, au1, vi1, lengths
                        )
                    txt_h0 = make_h0(av)
                    au_h0 = make_h0(tv)
                    vi_h0 = make_h0(ta)

        txt, au, vi = self._encode(
            txt, au, vi, lengths, txt_h0=txt_h0, au_h0=au_h0, vi_h0=vi_h0
        )
        fused, _ = self._fuse(txt, au, vi, lengths)

        return fused


class AudioVisualTextClassifier(nn.Module):
    def __init__(
        self,
        embeddings=None,
        vocab_size=None,
        audio_cfg=None,
        text_cfg=None,
        visual_cfg=None,
        fuse_cfg=None,
        modalities=None,
        text_mode="glove",
        num_classes=1,
        feedback=False,
        device="cpu",
    ):
        super(AudioVisualTextClassifier, self).__init__()
        self.modalities = modalities

        assert "text" in modalities or "glove" in modalities, "No text"
        assert "audio" in modalities, "No audio"
        assert "visual" in modalities, "No visual"

        self.encoder = AudioVisualTextEncoder(
            embeddings=embeddings,
            vocab_size=vocab_size,
            text_cfg=text_cfg,
            audio_cfg=audio_cfg,
            visual_cfg=visual_cfg,
            fuse_cfg=fuse_cfg,
            text_mode=text_mode,
            feedback=feedback,
            device=device,
        )

        self.classifier = nn.Linear(self.encoder.out_size, num_classes)

    def forward(self, inputs):
        out = self.encoder(
            inputs["text"], inputs["audio"], inputs["visual"], inputs["lengths"]
        )

        return self.classifier(out)
