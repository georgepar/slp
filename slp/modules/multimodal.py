import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from slp.modules.attention import Attention
from slp.modules.mmdrop import MultimodalDropout
from slp.modules.norm import LayerNorm


class MultiDropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(MultiDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError(
                f"dropout probability has to be between 0 and 1, " "but got {p}"
            )
        self.p = p
        self.inplace = inplace

    def forward(self, *mods):
        mods = list(mods)
        if self.training:
            for m in mods:
                m = F.dropout(m, p=self.p, inplace=self.inplace)

        return mods

    def __repr__(self):
        inplace_str = ", inplace" if self.inplace else ""
        return self.__class__.__name__ + "(" + "p=" + str(self.p) + inplace_str + ")"


class SymmetricAttention(nn.Module):
    """Some Information about Attention"""

    def __init__(
        self,
        attention_size=512,
        input_size=None,
        dropout=0.1,
        residual=1,
        layernorm=False,
    ):
        super(SymmetricAttention, self).__init__()

        if input_size is None:
            input_size = attention_size
        self.dk = input_size
        self.kx = nn.Linear(input_size, attention_size, bias=False)
        self.qx = nn.Linear(input_size, attention_size, bias=False)
        self.vx = nn.Linear(input_size, attention_size, bias=False)
        self.ky = nn.Linear(input_size, attention_size, bias=False)
        self.qy = nn.Linear(input_size, attention_size, bias=False)
        self.vy = nn.Linear(input_size, attention_size, bias=False)
        self.drop = nn.Dropout(dropout)
        self.layernorm = False

        if layernorm:
            self.layernorm = True
            self.lnx = LayerNorm(attention_size)
            self.lny = LayerNorm(attention_size)
        self.residual = residual
        self._reset_parameters()

    def forward(self, mod1, mod2, attention_mask=None):
        """
        x : (B, L, D)
        queries : (B, L, D)
        values : (B, L, D)
        """
        k_mod1 = self.kx(mod1)
        q_mod2 = self.qy(mod2)
        v_mod1 = self.vx(mod1)

        k_mod2 = self.ky(mod2)  # (B, L, A)
        q_mod1 = self.qx(mod1)
        v_mod2 = self.vy(mod2)

        # weights => (B, L, L)

        scores_mod1 = torch.bmm(q_mod2, k_mod1.transpose(1, 2)) / math.sqrt(self.dk)
        scores_mod2 = torch.bmm(q_mod1, k_mod2.transpose(1, 2)) / math.sqrt(self.dk)

        if attention_mask is not None:
            scores_mod1 = scores_mod1 + ((1 - attention_mask.unsqueeze(1)) * -1e5)
            scores_mod2 = scores_mod2 + ((1 - attention_mask.unsqueeze(1)) * -1e5)
        scores_mod1 = F.softmax(scores_mod1, dim=-1)
        scores_mod1 = self.drop(scores_mod1)
        scores_mod2 = F.softmax(scores_mod2, dim=-1)
        scores_mod2 = self.drop(scores_mod2)

        # out => (B, L, A)
        out_mod1 = torch.bmm(scores_mod1, v_mod1)
        out_mod2 = torch.bmm(scores_mod2, v_mod2)

        if self.layernorm:
            out_mod1 = self.lnx(out_mod1)
            out_mod2 = self.lny(out_mod2)

        if self.residual == 0:
            return out_mod1, out_mod2
        elif self.residual == 1:
            # vilbert cross residual

            # v + attention(v->a)
            # a + attention(a->v)
            out_mod1 += mod2
            out_mod2 += mod1

            return out_mod1, out_mod2
        elif self.residual == 2:
            out_mod1 += mod1
            out_mod2 += mod2

            return out_mod1, out_mod2

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.kx.weight)
        nn.init.xavier_uniform_(self.qx.weight)
        nn.init.xavier_uniform_(self.vx.weight)
        nn.init.xavier_uniform_(self.ky.weight)
        nn.init.xavier_uniform_(self.qy.weight)
        nn.init.xavier_uniform_(self.vy.weight)


class AttentionFuser(nn.Module):
    def __init__(
        self,
        proj_sz=None,
        residual=1,
        return_hidden=True,
        p_dropout=0.1,
        p_mmdrop=0.3,
        p_drop_modalities=None,
        multi_modal_drop="mmdrop_hard",
        mmdrop_before_fuse=True,
        mmdrop_after_fuse=True,
    ):
        super(AttentionFuser, self).__init__()
        self.return_hidden = return_hidden
        self.ta = SymmetricAttention(
            attention_size=proj_sz,
            dropout=p_dropout,
            residual=residual,
            layernorm=False,
        )

        self.va = SymmetricAttention(
            attention_size=proj_sz,
            dropout=p_dropout,
            residual=residual,
            layernorm=False,
        )

        self.tv = SymmetricAttention(
            attention_size=proj_sz,
            dropout=p_dropout,
            residual=residual,
            layernorm=False,
        )

        self.tav = Attention(
            attention_size=proj_sz,
            dropout=p_dropout,
        )

        self.mmdrop_before = None
        self.mmdrop_after = None

        if multi_modal_drop == "mmdrop_hard":
            if mmdrop_before_fuse:
                self.mmdrop_before = MultimodalDropout(
                    p=p_mmdrop,
                    n_modalities=3,
                    p_mod=p_drop_modalities,
                    mode="hard",
                )

            if mmdrop_after_fuse:
                self.mmdrop_after = MultimodalDropout(
                    p=p_mmdrop,
                    n_modalities=7,
                    p_mod=[1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7],
                    mode="hard",
                )

        elif multi_modal_drop == "mmdrop_soft":
            if mmdrop_before_fuse:
                self.mmdrop_before = MultimodalDropout(
                    p=p_mmdrop,
                    n_modalities=3,
                    p_mod=p_drop_modalities,
                    mode="soft",
                )

            if mmdrop_after_fuse:
                self.mmdrop_after = MultimodalDropout(
                    p=p_mmdrop,
                    n_modalities=7,
                    p_mod=[1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7],
                    mode="soft",
                )
        elif multi_modal_drop == "dropout":
            if mmdrop_before_fuse:
                # self.mmdrop_before = nn.Dropout(p_mmdrop)
                self.mmdrop_before = MultiDropout(p_mmdrop)

            if mmdrop_after_fuse:
                # self.mmdrop_after = nn.Dropout(p_mmdrop)
                self.mmdrop_after = MultiDropout(p_mmdrop)
        elif multi_modal_drop == "none":
            pass
        else:
            raise ValueError(
                "Not a specified mmdrop value given. Pls check your config file."
            )

        self.out_size = 7 * proj_sz

    def forward(self, txt, au, vi):
        if self.mmdrop_before is not None:
            txt, au, vi = self.mmdrop_before(txt, au, vi)
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

        if self.mmdrop_after is not None:
            txt, au, vi, ta, av, tv, tav = self.mmdrop_after(
                txt, au, vi, ta, av, tv, tav
            )

        # B x L x 7*D
        fused = torch.cat([txt, au, vi, ta, tv, av, tav], dim=-1)

        return fused
