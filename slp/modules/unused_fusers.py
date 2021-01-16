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
        elif mask_type == "attention":
            self.mask1 = Attention(
                attention_size=mod1_sz, query_size=hidden_dim, dropout=dropout
            )
            self.mask2 = Attention(
                attention_size=mod1_sz, query_size=hidden_dim, dropout=dropout
            )

            if use_self:
                self.mask_self = Attention(
                    attention_size=mod1_sz, query_size=hidden_dim, dropout=dropout
                )
        else:
            self.mask1 = nn.Linear(hidden_dim, mod1_sz)
            self.mask2 = nn.Linear(hidden_dim, mod1_sz)

            if use_self:
                self.mask_self = nn.Linear(hidden_dim, mod1_sz)

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

        ta = self.ta(txt, au)
        at = self.at(au, txt)
        av = self.av(au, vi)
        va = self.va(vi, au)
        vt = self.vt(vi, txt)
        tv = self.tv(txt, vi)

        av = va + av
        tv = vt + tv
        ta = ta + at

        tav = self.tav(txt, av)

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


