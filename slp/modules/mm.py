import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from slp.modules.attention import Attention, SymmetricAttention
from slp.modules.rnn import RNN, AttentiveRNN


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

            if self.p > 0:
                for m in range(len(mods)):
                    keep_prob = 1 - (self.p / self.n_modalities)  # (1 - self.p) * (self.n_modalities - 1) / self.n_modalities
                    mods[m] = mods[m] * (1 / keep_prob)

        return mods


class GatedMultimodalLayer(nn.Module):
    """ 
    Gated Multimodal Layer based on 
    'Gated multimodal networks, 
    Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) 
    """
    def __init__(self,
                 size_in1,
                 size_in2,
                 size_in3,
                 size_out=None):
        super(GatedMultimodalLayer, self).__init__()
        self.size_in1, self.size_in2, self.size_in3 = \
            size_in1, size_in2, size_in3
        self.size_out = size_out
        if self.size_out is None:
            self.size_out = size_in1

        self.hidden1 = nn.Linear(size_in1, size_out, bias=False)
        self.hidden2 = nn.Linear(size_in2, size_out, bias=False)
        self.hidden3 = nn.Linear(size_in3, size_out, bias=False)
        self.hidden_sigmoid1 = nn.Linear(size_out*3, 1, bias=False)
        self.hidden_sigmoid2 = nn.Linear(size_out*3, 1, bias=False)
        self.hidden_sigmoid3 = nn.Linear(size_out*3, 1, bias=False)

        # Activation functions
        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x1, x2):
        h1 = self.tanh_f(self.hidden1(x1))
        h2 = self.tanh_f(self.hidden2(x2))
        h3 = self.tanh_f(self.hidden3(x3))
        x = torch.cat((h1, h2, h3), dim=1)
        z1 = self.sigmoid_f(self.hidden_sigmoid1(x))
        z2 = self.sigmoid_f(self.hidden_sigmoid2(x))
        z3 = self.sigmoid_f(self.hidden_sigmoid3(x))

        # return z.view(z.size()[0],1)*h1 + (1-z).view(z.size()[0],1)*h2
        return z1.view(z1.size()[0],1)*h1 + z2.view(z2.size()[0],1)*h2 + z3.view(z3.size()[0],1)*h3


class FeedbackUnit(nn.Module):
    def __init__(
        self,
        hidden_dim1,
        hidden_dim2,
        mod_sz,
        use_self=False,
        mask_type="sigmoid",
        dropout=0.1,
        device="cpu",
        use_gmu=False
    ):
        super(FeedbackUnit, self).__init__()
        self.use_self = use_self
        self.mask_type = mask_type
        self.mod_sz = mod_sz
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.use_gmu = use_gmu

        if mask_type == "rnn" or mask_type == "sum_rnn":
            self.mask1 = RNN(hidden_dim1, mod_sz, dropout=dropout, device=device)
            self.mask2 = RNN(hidden_dim2, mod_sz, dropout=dropout, device=device)

            if self.use_gmu:
                pass
                # self.gmu = GatedMultimodalLayer()
            if use_self:
                self.mask_self = RNN(
                    hidden_dim1, mod_sz, dropout=dropout, device=device
                )
        elif mask_type == "attention":
            self.mask1 = Attention(
                attention_size=mod_sz, query_size=hidden_dim1, dropout=dropout
            )
            self.mask2 = Attention(
                attention_size=mod_sz, query_size=hidden_dim1, dropout=dropout
            )

            if use_self:
                self.mask_self = Attention(
                    attention_size=mod_sz, query_size=hidden_dim1, dropout=dropout
                )
        elif mask_type == "gmu":
            pass
        else:
            self.mask1 = nn.Linear(hidden_dim1, mod_sz)
            self.mask2 = nn.Linear(hidden_dim2, mod_sz)

            if use_self:
                self.mask_self = nn.Linear(hidden_dim1, mod_sz)

        mask_fn = {
            "sigmoid": self._sigmoid_mask,
            "sum_sigmoid": self._sum_sigmoid_mask,
            "softmax": self._softmax_mask,
            "sum_softmax": self._sum_softmax_mask,
            "dot": self._dot_mask,
            "rnn": self._rnn_mask,
            "sum_rnn": self._sum_rnn_mask,
            "attention": self._attention_mask,
        }

        self.get_mask = mask_fn[self.mask_type]

    def _attention_mask(self, x, y, z, x_high=None, lengths=None):
        _, m1 = self.mask1(x, queries=y)
        _, m2 = self.mask2(x, queries=z)

        mask = m1 + m2

        if self.use_self:
            _, m3 = self.mask_self(x_high)
            mask = mask + m3

        return mask

    def _sum_rnn_mask(self, x, y, z, x_high=None, lengths=None):
        oy, _, _ = self.mask1(y, lengths)
        oz, _, _ = self.mask2(z, lengths)

        if self.use_gmu:
            lg_y = self.gmu_y(x, oy)
            lg_z = self.gmu_z(x, oy)
            lg = (torch.sigmoid(lg_y) + torch.sigmoid(lg_z)) * 0.5
        else:
            lg = (torch.sigmoid(oy) + torch.sigmoid(oz)) * 0.5

        if self.use_self:
            ox, _, _ = self.mask_self(x_high, lengths)
            lg = lg + torch.sigmoid(ox)

        mask = lg

        return mask

    def _rnn_mask(self, x, y, z, x_high=None, lengths=None):
        oy, _, _ = self.mask1(y, lengths)
        oz, _, _ = self.mask2(z, lengths)

        lg = oy + oz

        if self.use_self:
            ox, _, _ = self.mask_self(x_high, lengths)
            lg = lg + ox

        mask = torch.sigmoid(lg)

        return mask

    def _sigmoid_mask(self, x, y, z, x_high=None, lengths=None):
        y = self.mask1(y)
        z = self.mask2(z)

        lg = y + z

        if self.use_self:
            m = self.mask_self(x_high)
            lg = lg + m

        mask = torch.sigmoid(lg)

        return mask

    def _softmax_mask(self, x, y, z, x_high=None, lengths=None):
        y = self.mask1(y)
        z = self.mask2(z)

        lg = y + z

        if self.use_self:
            m = self.mask_self(x_high)
            lg = lg + m

        mask = torch.softmax(lg, dim=-1)

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

    def _sum_softmax_mask(self, x, y, z, x_high=None, lengths=None):
        y = self.mask1(y)
        z = self.mask2(z)
        mask1 = torch.softmax(y, dim=-1)
        mask2 = torch.softmax(z, dim=-1)
        mask = mask1 + mask2

        if self.use_self:
            m = torch.softmax(self.mask_self(x_high), dim=-1)
            mask = mask + m

        return mask

    def _dot_mask(self, x, y, z, x_high=None, lengths=None):
        y = self.mask1(y)
        z = self.mask2(z)

        lg = y + z

        mask = torch.sigmoid(lg * x)

        return mask

    def forward(self, x, y, z, x_high=None, lengths=None):
        mask = self.get_mask(x, y, z, x_high=x_high, lengths=lengths)

        if self.mask_type == "attention":
            x = torch.bmm(mask, x)
        else:
            x = x * mask

        # if self.mod1_sz == 74:
        # print(mask[0][0])

        return x


class Feedback(nn.Module):
    def __init__(
        self,
        mod1_sz,
        mod2_sz,
        mod3_sz,
        mod1_hidden,
        mod2_hidden,
        mod3_hidden,
        use_self=False,
        mask_type="sigmoid",
        dropout=0.1,
        device="cpu",
    ):
        super(Feedback, self).__init__()
        self.f1 = FeedbackUnit(
            mod2_hidden,
            mod3_hidden,
            mod1_sz,
            use_self=use_self,
            mask_type=mask_type,
            dropout=dropout,
            device=device,
        )
        self.f2 = FeedbackUnit(
            mod1_hidden,
            mod3_hidden,
            mod2_sz,
            use_self=use_self,
            mask_type=mask_type,
            dropout=dropout,
            device=device,
        )
        self.f3 = FeedbackUnit(
            mod1_hidden,
            mod2_hidden,
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


class ModalityProjection(nn.Module):
    def __init__(self, mod1_sz, mod2_sz, mod3_sz, proj_sz):
        super(ModalityProjection, self).__init__()
        self.p = LinearProj(mod1_sz, mod2_sz, mod3_sz, proj_sz)

    def forward(self, x, y, z):
        x, y, z = self.p(x, y, z)

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
            self.proj = ModalityProjection(mod1_sz, mod2_sz, mod3_sz, self.proj_sz)

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


class AttentionFuser1(nn.Module):
    def __init__(
        self, proj_sz=None, residual=1, return_hidden=True, mmdrop=0, device="cpu"
    ):
        super(AttentionFuser1, self).__init__()
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

        self.vat = Attention(
            attention_size=proj_sz,
            dropout=0.1,
        )

        self.atv = Attention(
            attention_size=proj_sz,
            dropout=0.1,
        )

        self.mmdrop = MultimodalDropout(p=mmdrop, n_modalities=3, device=device)

        self.out_size = 9 * proj_sz

    def forward(self, txt, au, vi):
        txt, au, vi = self.mmdrop(txt, au, vi)
        ta, at = self.ta(txt, au)
        va, av = self.va(vi, au)
        tv, vt = self.tv(txt, vi)

        av = va + av
        tv = vt + tv
        ta = ta + at

        tav, _ = self.tav(txt, queries=av)
        vat, _ = self.vat(vi, queries=ta)
        atv, _ = self.atv(au, queries=tv)

        # Sum weighted attention hidden states

        if not self.return_hidden:
            txt = txt.sum(1)
            au = au.sum(1)
            vi = vi.sum(1)
            ta = ta.sum(1)
            av = av.sum(1)
            tv = tv.sum(1)
            tav = tav.sum(1)
            vat = vat.sum(1)
            atv = atv.sum(1)

        # B x L x 9*D
        fused = torch.cat([txt, au, vi, ta, tv, av, tav, vat, atv], dim=-1)

        return fused


class AttentionFuser(nn.Module):
    def __init__(
        self,
        a_hidden=None,
        t_hidden=None,
        v_hidden=None,
        proj_sz=None,
        residual=1,
        return_hidden=True,
        all_modalities=False,
        mmdrop=0,
        device="cpu"
    ):
        super(AttentionFuser, self).__init__()
        self.return_hidden = return_hidden
        self.all_modalities = all_modalities
        self.ta = SymmetricAttention(
            mod1_size=t_hidden,
            mod2_size=a_hidden,
            attention_size=proj_sz,
            dropout=0.1,
            residual=residual,
            layernorm=False,
        )

        self.va = SymmetricAttention(
            mod1_size=v_hidden,
            mod2_size=a_hidden,
            attention_size=proj_sz,
            dropout=0.1,
            residual=residual,
            layernorm=False,
        )

        self.tv = SymmetricAttention(
            mod1_size=t_hidden,
            mod2_size=v_hidden,
            attention_size=proj_sz,
            dropout=0.1,
            residual=residual,
            layernorm=False,
        )

        self.tav = Attention(
            input_size=t_hidden,
            query_size=proj_sz,
            attention_size=proj_sz,
            dropout=0.1,
        )

        self.out_size = a_hidden + v_hidden + t_hidden + 4*proj_sz

        if self.all_modalities:
            self.atv = Attention(
                input_size=a_hidden,
                query_size=proj_sz,
                attention_size=proj_sz,
                dropout=0.1,
            )

            self.vat = Attention(
                input_size=v_hidden,
                query_size=proj_sz,
                attention_size=proj_sz,
                dropout=0.1,
            )

            self.out_size = a_hidden + v_hidden + t_hidden + 6*proj_sz

        self.mmdrop = MultimodalDropout(p=mmdrop, n_modalities=3, device=device)


    def forward(self, txt, au, vi):
        txt, au, vi = self.mmdrop(txt, au, vi)
        # print(f"{txt.size()}")
        ta, at = self.ta(txt, au)
        va, av = self.va(vi, au)
        tv, vt = self.tv(txt, vi)
        # print(f"hello")
        av = va + av
        tv = vt + tv
        ta = ta + at

        tav, _ = self.tav(txt, queries=av)
        if self.all_modalities:
            atv, _ = self.atv(au, queries=tv)
            vat, _ = self.vat(vi, queries=ta)

        # print(f"hello3")
        # Sum weighted attention hidden states

        if not self.return_hidden:
            txt = txt.sum(1)
            au = au.sum(1)
            vi = vi.sum(1)
            ta = ta.sum(1)
            av = av.sum(1)
            tv = tv.sum(1)
            tav = tav.sum(1)
            if self.all_modalities:
                atv = atv.sum(1)
                vat = vat.sum(1)

        # B x L x 7*D
        if self.all_modalities:
            fused = torch.cat([txt, au, vi, ta, tv, av, tav, atv, vat], dim=-1)
        else:
            fused = torch.cat([txt, au, vi, ta, tv, av, tav], dim=-1)

        return fused



class BiAttentionFuser(nn.Module):
    def __init__(
        self,
        mod1_hidden=None,
        mod2_hidden=None,
        proj_sz=None,
        residual=1,
        return_hidden=True,
        all_modalities=False,
        mmdrop=0,
        device="cpu"
    ):
        super(BiAttentionFuser, self).__init__()
        self.return_hidden = return_hidden
        self.all_modalities = all_modalities
        self.cross_mod = SymmetricAttention(
            mod1_size=mod1_hidden,
            mod2_size=mod2_hidden,
            attention_size=proj_sz,
            dropout=0.1,
            residual=residual,
            layernorm=False,
        )

        self.out_size = mod1_hidden + mod2_hidden + proj_sz

        self.mmdrop = \
            MultimodalDropout(p=mmdrop, n_modalities=3, device=device)


    def forward(self, mod_1, mod_2):
        mod_1, mod_2 = self.mmdrop(mod_1, mod_2)
        mod_12, mod_21 = self.cross_mod(mod_1, mod_2)
        mod_cross = mod_12 + mod_21

        if not self.return_hidden:
            mod_1 = mod_1.sum(1)
            mod_2 = mod_2.sum(1)
            mod_cross = mod_cross.sum(1)

        fused = torch.cat([mod_1, mod_2, mod_cross], dim=-1)

        return fused


class BilinearFuser(nn.Module):
    def __init__(self, proj_sz=None, mmdrop=0, device="cpu", return_hidden=False):
        super(BilinearFuser, self).__init__()
        self.return_hidden = return_hidden
        self.ta = nn.Bilinear(proj_sz, proj_sz, proj_sz)
        self.at = nn.Bilinear(proj_sz, proj_sz, proj_sz)
        self.va = nn.Bilinear(proj_sz, proj_sz, proj_sz)
        self.av = nn.Bilinear(proj_sz, proj_sz, proj_sz)
        self.tv = nn.Bilinear(proj_sz, proj_sz, proj_sz)
        self.vt = nn.Bilinear(proj_sz, proj_sz, proj_sz)
        self.tav = nn.Bilinear(proj_sz, proj_sz, proj_sz)
        self.drop = nn.Dropout(0.2)

        self.mmdrop = MultimodalDropout(p=mmdrop, n_modalities=3, device=device)

        self.out_size = 7 * proj_sz

    def forward(self, txt, au, vi):
        txt, au, vi = self.mmdrop(txt, au, vi)
        # print("hello")
        ta = self.ta(txt, au)
        at = self.at(au, txt)
        av = self.av(au, vi)
        va = self.va(vi, au)
        vt = self.vt(vi, txt)
        tv = self.tv(txt, vi)
        # print("hello2")
        av = va + av
        tv = vt + tv
        ta = ta + at
        # print("hello3")
        tav = self.tav(txt, av)
        # print("hello4")
        if not self.return_hidden:
            txt = txt.sum(1)
            au = au.sum(1)
            vi = vi.sum(1)
            ta = ta.sum(1)
            av = av.sum(1)
            tv = tv.sum(1)
            tav = tav.sum(1)

        fused = torch.cat([txt, au, vi, ta, tv, av, tav], dim=-1)

        return fused


class BilinearRnnFuser(nn.Module):
    def __init__(
        self,
        proj_sz=None,
        mmdrop=0,
        device="cpu",
    ):
        super(BilinearRnnFuser, self).__init__()
        self.att_fuser = BilinearFuser(
            proj_sz=proj_sz,
            return_hidden=True,
            mmdrop=mmdrop,
            device=device,
        )
        self.rnn = AttentiveRNN(
            self.att_fuser.out_size,
            proj_sz,
            bidirectional=True,
            merge_bi="cat",
            attention=True,
            device=device,
        )
        self.out_size = self.rnn.out_size

    def forward(self, txt, au, vi, lengths):
        att = self.att_fuser(txt, au, vi)
        out = self.rnn(att, lengths)

        return out


class AttRnnFuser(nn.Module):
    def __init__(
        self,
        a_hidden=None,
        t_hidden=None,
        v_hidden=None,
        proj_sz=None,
        layers=1,
        residual=1,
        mmdrop=0,
        device="cpu",
        return_hidden=False,
        all_modalities=False
    ):
        super(AttRnnFuser, self).__init__()
        self.att_fuser = AttentionFuser(
            a_hidden=a_hidden,
            t_hidden=t_hidden,
            v_hidden=v_hidden,
            proj_sz=proj_sz,
            residual=residual,
            return_hidden=True,
            all_modalities=all_modalities,
            mmdrop=mmdrop,
            device=device,
        )
        self.rnn = AttentiveRNN(
            self.att_fuser.out_size,
            proj_sz,
            bidirectional=True,
            layers=layers,
            merge_bi="cat",
            attention=True,
            device=device,
            return_hidden=return_hidden,
        )
        self.out_size = self.rnn.out_size

    def forward(self, txt, au, vi, lengths):
        att = self.att_fuser(txt, au, vi)  # B x L x 7 * D
        # print(f"attention size is {att.size()}")
        out = self.rnn(att, lengths)  # B x L x 2 * D

        return out



class BiAttRnnFuser(nn.Module):
    def __init__(
        self,
        mod1_hidden=None,
        mod2_hidden=None,
        proj_sz=None,
        layers=1,
        residual=1,
        mmdrop=0,
        device="cpu",
        return_hidden=False,
        all_modalities=False
    ):
        super(BiAttRnnFuser, self).__init__()
        self.att_fuser = BiAttentionFuser(
            mod1_hidden=mod1_hidden,
            mod2_hidden=mod2_hidden,
            proj_sz=proj_sz,
            residual=residual,
            return_hidden=True,
            all_modalities=all_modalities,
            mmdrop=mmdrop,
            device=device,
        )
        self.rnn = AttentiveRNN(
            self.att_fuser.out_size,
            proj_sz,
            bidirectional=True,
            layers=layers,
            merge_bi="cat",
            attention=True,
            device=device,
            return_hidden=return_hidden,
        )
        self.out_size = self.rnn.out_size

    def forward(self, mod1, mod2, lengths):
        att = self.att_fuser(mod1, mod2)  # B x L x 7 * D
        # print(f"attention size is {att.size()}")
        out = self.rnn(att, lengths)  # B x L x 2 * D

        return out


class AttRnnFuser1(nn.Module):
    def __init__(
        self,
        proj_sz=None,
        residual=1,
        mmdrop=0,
        device="cpu",
    ):
        super(AttRnnFuser1, self).__init__()
        self.att_fuser = AttentionFuser1(
            proj_sz=proj_sz,
            residual=residual,
            return_hidden=True,
            mmdrop=mmdrop,
            device=device,
        )
        self.rnn = AttentiveRNN(
            self.att_fuser.out_size,
            proj_sz,
            bidirectional=True,
            merge_bi="cat",
            attention=True,
            device=device,
        )
        self.out_size = self.rnn.out_size

    def forward(self, txt, au, vi, lengths):
        att = self.att_fuser(txt, au, vi)  # B x L x 9 * D
        out = self.rnn(att, lengths)  # B x L x 2 * D

        return out


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

    def forward(self, x, lengths):
        if self.bn is not None:
            x = self.bn(x.view(-1, x.size(2), x.size(1))).view(-1, x.size(1), x.size(2))
        x = self.audio(x, lengths)

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

    def forward(self, x, lengths):
        if self.bn is not None:
            x = self.bn(x.view(-1, x.size(2), x.size(1))).view(-1, x.size(1), x.size(2))
        x = self.visual(x, lengths)

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

    def forward(self, x, lengths):
        x = self.text(x, lengths)

        return x


class GloveClassifier(nn.Module):
    def __init__(self, cfg,
                 num_classes=1,
                 project=False,
                 projection_size=300,
                 device="cpu"):
        super(GloveClassifier, self).__init__()
        self.proj = None
        if (cfg["hidden_size"] != projection_size) and project:
            self.proj = nn.Linear(cfg["input_size"], projection_size)
        self.encoder = GloveEncoder(cfg, device=device)
        self.classifier = nn.Linear(self.encoder.out_size, num_classes)

    def forward(self, x, lengths):
        if self.proj is not None:
            x = self.proj(x)
        x = self.encoder(x, lengths)
        return self.classifier(x)

class AudioClassifier(nn.Module):
    def __init__(self, cfg,
                 num_classes=1,
                 project=False,
                 projection_size=300,
                 device="cpu"):
        super(AudioClassifier, self).__init__()
        self.proj = None
        if (cfg["hidden_size"] != projection_size) and project:
            self.proj = nn.Linear(cfg["input_size"], projection_size)
        self.encoder = AudioEncoder(cfg, device=device)
        self.classifier = nn.Linear(self.encoder.out_size, num_classes)

    def forward(self, x, lengths):
        if self.proj is not None:
            x = self.proj(x)
        x = self.encoder(x, lengths)
        return self.classifier(x)

class VisualClassifier(nn.Module):
    def __init__(self, cfg,
                 num_classes=1,
                 project=False,
                 projection_size=300,
                 device="cpu"):
        super(VisualClassifier, self).__init__()
        self.proj = None
        if (cfg["hidden_size"] != projection_size) and project:
            self.proj = nn.Linear(cfg["input_size"], projection_size)
        self.encoder = VisualEncoder(cfg, device=device)
        self.classifier = nn.Linear(self.encoder.out_size, num_classes)

    def forward(self, x, lengths):
        if self.proj is not None:
            x = self.proj(x)
        x = self.encoder(x, lengths)
        return self.classifier(x)


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

        a_hidden = audio_cfg["hidden_size"]
        t_hidden = text_cfg["hidden_size"]
        v_hidden = visual_cfg["hidden_size"]

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

        elif fuse_cfg["method"] == "common_space":
            raise NotImplementedError
        elif fuse_cfg["method"] == "att":
            self.fuser = AttentionFuser(
                proj_sz=fuse_cfg["projection_size"],
                residual=fuse_cfg["residual"],
                mmdrop=fuse_cfg["mmdrop"],
                return_hidden=False,
                device=device,
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
                a_hidden=a_hidden,
                t_hidden=t_hidden,
                v_hidden=v_hidden,
                proj_sz=fuse_cfg["projection_size"],
                residual=fuse_cfg["residual"],
                mmdrop=fuse_cfg["mmdrop"],
                all_modalities=fuse_cfg["all_modalities"],
                layers=fuse_cfg["layers"],
                device=device,
            )
        elif fuse_cfg["method"] == "attrnn1":
            self.fuser = AttRnnFuser1(
                proj_sz=fuse_cfg["projection_size"],
                residual=fuse_cfg["residual"],
                mmdrop=fuse_cfg["mmdrop"],
                device=device,
            )
        elif fuse_cfg["method"] == "bilinear":
            self.fuser = BilinearFuser(
                proj_sz=fuse_cfg["projection_size"],
                mmdrop=fuse_cfg["mmdrop"],
                return_hidden=False,
                device=device,
            )
        elif fuse_cfg["method"] == "birnn":
            self.fuser = BilinearRnnFuser(
                proj_sz=fuse_cfg["projection_size"],
                mmdrop=fuse_cfg["mmdrop"],
                device=device,
            )
        else:
            raise ValueError('Supported fuse techniques: ["cat", "add"]')

        self.fuse_method = fuse_cfg["method"]

        self.out_size = self.fuser.out_size

        if feedback:
            self.fm = Feedback(
                mod1_sz=text_cfg["orig_size"],
                mod2_sz=audio_cfg["orig_size"],
                mod3_sz=visual_cfg["orig_size"],
                mod1_hidden=text_cfg["hidden_size"],
                mod2_hidden=audio_cfg["hidden_size"],
                mod3_hidden=visual_cfg["hidden_size"],
                use_self=fuse_cfg["self_feedback"],
                mask_type=fuse_cfg["feedback_type"],
                dropout=0.1,
                device=device,
            )

    def _encode(self, txt, au, vi, lengths):
        if self.proj is not None:
            txt, au, vi = self.proj(txt, au, vi)
        txt = self.text(txt, lengths)
        au = self.audio(au, lengths)
        vi = self.visual(vi, lengths)

        return txt, au, vi

    def _fuse(self, txt, au, vi, lengths):
        if self.fuse_method in ["cat", "sum"]:
            # Sum weighted attention hidden states
            fused = self.fuser(txt.sum(1), au.sum(1), vi.sum(1))
        elif self.fuse_method in ["att", "bilinear"]:
            fused = self.fuser(txt, au, vi)
        else:
            fused = self.fuser(txt, au, vi, lengths)

        return fused

    def forward(self, txt, au, vi, lengths):
        if self.feedback:
            for _ in range(1):
                txt1, au1, vi1 = self._encode(txt, au, vi, lengths)
                # print(f"audio size is {au1.size()}")
                # print(f"text size is {txt1.size()}")
                # print(f"video size is {vi1.size()}")
                txt, au, vi = self.fm(txt, au, vi, txt1, au1, vi1, lengths=lengths)

        txt, au, vi = self._encode(txt, au, vi, lengths)
        # print(f"audio size is {au.size()}")
        fused = self._fuse(txt, au, vi, lengths)

        return fused


class AudioVisualEncoder(nn.Module):
    def __init__(
        self,
        audio_cfg=None,
        visual_cfg=None,
        fuse_cfg=None,
        feedback=False,
        device="cpu",
    ):
        super(AudioVisualEncoder, self).__init__()
        # For now model dim == text dim (the largest). In the future this can be done
        # with individual projection layers for each modality
        assert (
            audio_cfg["attention"] and visual_cfg["attention"]
        ), "Use attention pls."

        self.feedback = feedback
        audio_cfg["orig_size"] = audio_cfg.get("orig_size", audio_cfg["input_size"])
        visual_cfg["orig_size"] = visual_cfg.get("orig_size", visual_cfg["input_size"])

        self.proj = None
        if self.proj is not None:
            audio_cfg["input_size"] = fuse_cfg["model_dim"]
            visual_cfg["input_size"] = fuse_cfg["model_dim"]

        audio_cfg["return_hidden"] = True
        visual_cfg["return_hidden"] = True

        a_hidden = audio_cfg["hidden_size"]
        v_hidden = visual_cfg["hidden_size"]


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

        elif fuse_cfg["method"] == "common_space":
            raise NotImplementedError
        elif fuse_cfg["method"] == "att":
            self.fuser = AttentionFuser(
                proj_sz=fuse_cfg["projection_size"],
                residual=fuse_cfg["residual"],
                mmdrop=fuse_cfg["mmdrop"],
                return_hidden=False,
                device=device,
            )
        elif fuse_cfg["method"] == "rnn":
            self.fuser = RnnFuser(
                [self.text.out_size, self.audio.out_size, self.visual.out_size],
                proj_sz=fuse_cfg["projection_size"],
                mmdrop=fuse_cfg["mmdrop"],
                device=device,
            )
        elif fuse_cfg["method"] == "attrnn":
            # curtrently this is the only supported fusion method for bimodal
            self.fuser = BiAttRnnFuser(
                mod1_hidden=a_hidden,
                mod2_hidden=v_hidden,
                proj_sz=fuse_cfg["projection_size"],
                residual=fuse_cfg["residual"],
                mmdrop=fuse_cfg["mmdrop"],
                all_modalities=fuse_cfg["all_modalities"],
                layers=fuse_cfg["layers"],
                device=device,
            )
        elif fuse_cfg["method"] == "attrnn1":
            self.fuser = AttRnnFuser1(
                proj_sz=fuse_cfg["projection_size"],
                residual=fuse_cfg["residual"],
                mmdrop=fuse_cfg["mmdrop"],
                device=device,
            )
        elif fuse_cfg["method"] == "bilinear":
            self.fuser = BilinearFuser(
                proj_sz=fuse_cfg["projection_size"],
                mmdrop=fuse_cfg["mmdrop"],
                return_hidden=False,
                device=device,
            )
        elif fuse_cfg["method"] == "birnn":
            self.fuser = BilinearRnnFuser(
                proj_sz=fuse_cfg["projection_size"],
                mmdrop=fuse_cfg["mmdrop"],
                device=device,
            )
        else:
            raise ValueError('Supported fuse techniques: ["cat", "add"]')

        self.fuse_method = fuse_cfg["method"]

        self.out_size = self.fuser.out_size

        #  TODO for bimodal
        # if feedback:
        #     self.fm = Feedback(
        #         mod1_sz=text_cfg["orig_size"],
        #         mod2_sz=audio_cfg["orig_size"],
        #         mod3_sz=visual_cfg["orig_size"],
        #         mod1_hidden=text_cfg["hidden_size"],
        #         mod2_hidden=audio_cfg["hidden_size"],
        #         mod3_hidden=visual_cfg["hidden_size"],
        #         use_self=fuse_cfg["self_feedback"],
        #         mask_type=fuse_cfg["feedback_type"],
        #         dropout=0.1,
        #         device=device,
        #     )

    def _encode(self, au, vi, lengths):
        if self.proj is not None:
            au, vi = self.proj(au, vi)
        au = self.audio(au, lengths)
        vi = self.visual(vi, lengths)

        return au, vi

    def _fuse(self, au, vi, lengths):
        if self.fuse_method in ["cat", "sum"]:
            # Sum weighted attention hidden states
            fused = self.fuser(au.sum(1), vi.sum(1))
        elif self.fuse_method in ["att", "bilinear"]:
            fused = self.fuser(au, vi)
        else:
            fused = self.fuser(au, vi, lengths)

        return fused

    def forward(self, au, vi, lengths):
        if self.feedback:
            for _ in range(1):
                au1, vi1 = self._encode(au, vi, lengths)
                # print(f"audio size is {au1.size()}")
                # print(f"text size is {txt1.size()}")
                # print(f"video size is {vi1.size()}")
                au, vi = self.fm(au, vi, au1, vi1, lengths=lengths)

        au, vi = self._encode(au, vi, lengths)
        # print(f"audio size is {au.size()}")
        fused = self._fuse(au, vi, lengths)

        return fused


class AudioTextEncoder(nn.Module):
    def __init__(
        self,
        embeddings=None,
        vocab_size=None,
        audio_cfg=None,
        text_cfg=None,
        fuse_cfg=None,
        feedback=False,
        text_mode="glove",
        device="cpu",
    ):
        super(AudioTextEncoder, self).__init__()
        # For now model dim == text dim (the largest). In the future this can be done
        # with individual projection layers for each modality
        assert (
            text_cfg["attention"] and audio_cfg["attention"]
        ), "Use attention pls."

        self.feedback = feedback
        text_cfg["orig_size"] = text_cfg.get("orig_size", text_cfg["input_size"])
        audio_cfg["orig_size"] = audio_cfg.get("orig_size", audio_cfg["input_size"])
        
        if fuse_cfg["projection_type"] == "conv":
            self.proj = Conv1dProj(
                text_cfg["orig_size"],
                audio_cfg["orig_size"],
                fuse_cfg["model_dim"],
            )
        elif fuse_cfg["projection_type"] == "linear":
            self.proj = LinearProj(
                text_cfg["orig_size"],
                audio_cfg["orig_size"],
                fuse_cfg["model_dim"],
            )
        else:
            self.proj = None

        if self.proj is not None:
            text_cfg["input_size"] = fuse_cfg["model_dim"]
            audio_cfg["input_size"] = fuse_cfg["model_dim"]
        
        text_cfg["return_hidden"] = True
        audio_cfg["return_hidden"] = True
        
        a_hidden = audio_cfg["hidden_size"]
        t_hidden = text_cfg["hidden_size"]
        
        if text_mode == "glove":
            self.text = GloveEncoder(text_cfg, device=device)
        else:
            raise ValueError("Only glove supported for now")

        self.audio = AudioEncoder(audio_cfg, device=device)

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

        elif fuse_cfg["method"] == "common_space":
            raise NotImplementedError
        elif fuse_cfg["method"] == "att":
            self.fuser = AttentionFuser(
                proj_sz=fuse_cfg["projection_size"],
                residual=fuse_cfg["residual"],
                mmdrop=fuse_cfg["mmdrop"],
                return_hidden=False,
                device=device,
            )
        elif fuse_cfg["method"] == "rnn":
            self.fuser = RnnFuser(
                [self.text.out_size, self.audio.out_size, self.visual.out_size],
                proj_sz=fuse_cfg["projection_size"],
                mmdrop=fuse_cfg["mmdrop"],
                device=device,
            )
        elif fuse_cfg["method"] == "attrnn":
            self.fuser = BiAttRnnFuser(
                mod1_hidden=a_hidden,
                mod2_hidden=t_hidden,
                proj_sz=fuse_cfg["projection_size"],
                residual=fuse_cfg["residual"],
                mmdrop=fuse_cfg["mmdrop"],
                all_modalities=fuse_cfg["all_modalities"],
                layers=fuse_cfg["layers"],
                device=device,
            )
        elif fuse_cfg["method"] == "attrnn1":
            self.fuser = AttRnnFuser1(
                proj_sz=fuse_cfg["projection_size"],
                residual=fuse_cfg["residual"],
                mmdrop=fuse_cfg["mmdrop"],
                device=device,
            )
        elif fuse_cfg["method"] == "bilinear":
            self.fuser = BilinearFuser(
                proj_sz=fuse_cfg["projection_size"],
                mmdrop=fuse_cfg["mmdrop"],
                return_hidden=False,
                device=device,
            )
        elif fuse_cfg["method"] == "birnn":
            self.fuser = BilinearRnnFuser(
                proj_sz=fuse_cfg["projection_size"],
                mmdrop=fuse_cfg["mmdrop"],
                device=device,
            )
        else:
            raise ValueError('Supported fuse techniques: ["cat", "add"]')

        self.fuse_method = fuse_cfg["method"]

        self.out_size = self.fuser.out_size

        # if feedback:
        #     self.fm = Feedback(
        #         mod1_sz=text_cfg["orig_size"],
        #         mod2_sz=audio_cfg["orig_size"],
        #         mod3_sz=visual_cfg["orig_size"],
        #         mod1_hidden=text_cfg["hidden_size"],
        #         mod2_hidden=audio_cfg["hidden_size"],
        #         mod3_hidden=visual_cfg["hidden_size"],
        #         use_self=fuse_cfg["self_feedback"],
        #         mask_type=fuse_cfg["feedback_type"],
        #         dropout=0.1,
        #         device=device,
        #     )

    def _encode(self, au, txt, lengths):
        if self.proj is not None:
            txt, au = self.proj(txt, au)
        txt = self.text(txt, lengths)
        au = self.audio(au, lengths)

        return au, txt

    def _fuse(self, au, txt, lengths):
        if self.fuse_method in ["cat", "sum"]:
            # Sum weighted attention hidden states
            fused = self.fuser(au.sum(1), txt.sum(1))
        elif self.fuse_method in ["att", "bilinear"]:
            fused = self.fuser(au, txt)
        else:
            fused = self.fuser(au, txt, lengths)

        return fused

    def forward(self, au, txt, lengths):
        if self.feedback:
            for _ in range(1):
                au1, txt1 = self._encode(au, txt, lengths)
                # print(f"audio size is {au1.size()}")
                # print(f"text size is {txt1.size()}")
                # print(f"video size is {vi1.size()}")
                au, txt = self.fm(au, txt, au1, txt1, lengths=lengths)

        au, txt = self._encode(au, txt, lengths)
        # print(f"audio size is {au.size()}")
        fused = self._fuse(au, txt, lengths)

        return fused


class VisualTextEncoder(nn.Module):
    def __init__(
        self,
        embeddings=None,
        vocab_size=None,
        text_cfg=None,
        visual_cfg=None,
        fuse_cfg=None,
        feedback=False,
        text_mode="glove",
        device="cpu",
    ):
        super(VisualTextEncoder, self).__init__()
        # For now model dim == text dim (the largest). In the future this can be done
        # with individual projection layers for each modality
        assert (
            text_cfg["attention"] and visual_cfg["attention"]
        ), "Use attention pls."

        self.feedback = feedback
        text_cfg["orig_size"] = text_cfg.get("orig_size", text_cfg["input_size"])
        visual_cfg["orig_size"] = visual_cfg.get("orig_size", visual_cfg["input_size"])

        if fuse_cfg["projection_type"] == "conv":
            self.proj = Conv1dProj(
                text_cfg["orig_size"],
                visual_cfg["orig_size"],
                fuse_cfg["model_dim"],
            )
        elif fuse_cfg["projection_type"] == "linear":
            self.proj = LinearProj(
                text_cfg["orig_size"],
                visual_cfg["orig_size"],
                fuse_cfg["model_dim"],
            )
        else:
            self.proj = None

        if self.proj is not None:
            text_cfg["input_size"] = fuse_cfg["model_dim"]
            visual_cfg["input_size"] = fuse_cfg["model_dim"]

        text_cfg["return_hidden"] = True
        visual_cfg["return_hidden"] = True

        t_hidden = text_cfg["hidden_size"]
        v_hidden = visual_cfg["hidden_size"]

        if text_mode == "glove":
            self.text = GloveEncoder(text_cfg, device=device)
        else:
            raise ValueError("Only glove supported for now")

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

        elif fuse_cfg["method"] == "common_space":
            raise NotImplementedError
        elif fuse_cfg["method"] == "att":
            self.fuser = AttentionFuser(
                proj_sz=fuse_cfg["projection_size"],
                residual=fuse_cfg["residual"],
                mmdrop=fuse_cfg["mmdrop"],
                return_hidden=False,
                device=device,
            )
        elif fuse_cfg["method"] == "rnn":
            self.fuser = RnnFuser(
                [self.text.out_size, self.audio.out_size, self.visual.out_size],
                proj_sz=fuse_cfg["projection_size"],
                mmdrop=fuse_cfg["mmdrop"],
                device=device,
            )
        elif fuse_cfg["method"] == "attrnn":
            self.fuser = BiAttRnnFuser(
                mod1_hidden=v_hidden,
                mod2_hidden=t_hidden,
                proj_sz=fuse_cfg["projection_size"],
                residual=fuse_cfg["residual"],
                mmdrop=fuse_cfg["mmdrop"],
                all_modalities=fuse_cfg["all_modalities"],
                layers=fuse_cfg["layers"],
                device=device,
            )
        elif fuse_cfg["method"] == "attrnn1":
            self.fuser = AttRnnFuser1(
                proj_sz=fuse_cfg["projection_size"],
                residual=fuse_cfg["residual"],
                mmdrop=fuse_cfg["mmdrop"],
                device=device,
            )
        elif fuse_cfg["method"] == "bilinear":
            self.fuser = BilinearFuser(
                proj_sz=fuse_cfg["projection_size"],
                mmdrop=fuse_cfg["mmdrop"],
                return_hidden=False,
                device=device,
            )
        elif fuse_cfg["method"] == "birnn":
            self.fuser = BilinearRnnFuser(
                proj_sz=fuse_cfg["projection_size"],
                mmdrop=fuse_cfg["mmdrop"],
                device=device,
            )
        else:
            raise ValueError('Supported fuse techniques: ["cat", "add"]')

        self.fuse_method = fuse_cfg["method"]

        self.out_size = self.fuser.out_size

        # if feedback:
        #     self.fm = Feedback(
        #         mod1_sz=text_cfg["orig_size"],
        #         mod2_sz=audio_cfg["orig_size"],
        #         mod3_sz=visual_cfg["orig_size"],
        #         mod1_hidden=text_cfg["hidden_size"],
        #         mod2_hidden=audio_cfg["hidden_size"],
        #         mod3_hidden=visual_cfg["hidden_size"],
        #         use_self=fuse_cfg["self_feedback"],
        #         mask_type=fuse_cfg["feedback_type"],
        #         dropout=0.1,
        #         device=device,
        #     )

    def _encode(self, vi, txt, lengths):
        if self.proj is not None:
            vi, txt = self.proj(vi, txt)
        txt = self.text(txt, lengths)
        vi = self.visual(vi, lengths)

        return vi, txt

    def _fuse(self, vi, txt, lengths):
        if self.fuse_method in ["cat", "sum"]:
            # Sum weighted attention hidden states
            fused = self.fuser(vi.sum(1), txt.sum(1))
        elif self.fuse_method in ["att", "bilinear"]:
            fused = self.fuser(vi, txt)
        else:
            fused = self.fuser(vi, txt, lengths)

        return fused

    def forward(self, vi, txt, lengths):
        if self.feedback:
            for _ in range(1):
                vi1, txt1 = self._encode(vi, txt, lengths)
                # print(f"audio size is {au1.size()}")
                # print(f"text size is {txt1.size()}")
                # print(f"video size is {vi1.size()}")
                vi, txt = self.fm(vi, txt, vi1, txt1, lengths=lengths)

        vi, txt = self._encode(vi, txt, lengths)
        # print(f"audio size is {au.size()}")
        fused = self._fuse(vi, txt, lengths)

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


class AudioVisualClassifier(nn.Module):
    def __init__(
        self,
        audio_cfg=None,
        visual_cfg=None,
        fuse_cfg=None,
        modalities=None,
        num_classes=1,
        feedback=False,
        device="cpu",
    ):
        super(AudioVisualClassifier, self).__init__()
        self.modalities = modalities

        assert "audio" in modalities, "No audio"
        assert "visual" in modalities, "No visual"

        self.encoder = AudioVisualEncoder(
            audio_cfg=audio_cfg,
            visual_cfg=visual_cfg,
            fuse_cfg=fuse_cfg,
            feedback=feedback,
            device=device,
        )

        self.classifier = nn.Linear(self.encoder.out_size, num_classes)

    def forward(self, au, vi, lengths):
        out = self.encoder(au, vi, lengths)

        return self.classifier(out)


class AudioTextClassifier(nn.Module):
    def __init__(
        self,
        embeddings=None,
        vocab_size=None,
        audio_cfg=None,
        text_cfg=None,
        fuse_cfg=None,
        modalities=None,
        text_mode="glove",
        num_classes=1,
        feedback=False,
        device="cpu",
    ):
        super(AudioTextClassifier, self).__init__()
        self.modalities = modalities

        assert "text" in modalities or "glove" in modalities, "No text"
        assert "audio" in modalities, "No audio"

        self.encoder = AudioTextEncoder(
            embeddings=embeddings,
            vocab_size=vocab_size,
            text_cfg=text_cfg,
            audio_cfg=audio_cfg,
            fuse_cfg=fuse_cfg,
            text_mode=text_mode,
            feedback=feedback,
            device=device,
        )

        self.classifier = nn.Linear(self.encoder.out_size, num_classes)

    def forward(self, au, txt, lengths):
        out = self.encoder(au, txt, lengths)

        return self.classifier(out)


class VisualTextClassifier(nn.Module):
    def __init__(
        self,
        embeddings=None,
        vocab_size=None,
        text_cfg=None,
        visual_cfg=None,
        fuse_cfg=None,
        modalities=None,
        text_mode="glove",
        num_classes=1,
        feedback=False,
        device="cpu",
    ):
        super(VisualTextClassifier, self).__init__()
        self.modalities = modalities

        assert "text" in modalities or "glove" in modalities, "No text"
        assert "visual" in modalities, "No visual"

        self.encoder = VisualTextEncoder(
            embeddings=embeddings,
            vocab_size=vocab_size,
            text_cfg=text_cfg,
            visual_cfg=visual_cfg,
            fuse_cfg=fuse_cfg,
            text_mode=text_mode,
            feedback=feedback,
            device=device,
        )

        self.classifier = nn.Linear(self.encoder.out_size, num_classes)

    def forward(self, vi, txt, lengths):
        out = self.encoder(vi, txt, lengths)

        return self.classifier(out)
