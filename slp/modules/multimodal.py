import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from slp.modules.rnn import AttentiveRNN, WordRNN, CoAttentiveRNN


class GatedLinearUnit(nn.Module):
    def __init__(self, hidden_dim=None, learnable=False):
        super(GatedLinearUnit, self).__init__()
        self.learnable = learnable

        if learnable:
            if hidden_dim is None:
                raise ValueError("You must provide hidden dim for learnable GLU")
            self.hidden_dim = hidden_dim
            self.proj = nn.Linear(hidden_dim, hidden_dim)
            self.mask = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, y):
        if self.learnable:
            x = self.proj(x)
            y = self.mask(y)
        mask = torch.sigmoid(y)
        x = x * mask

        return x


class GatedLinearUnit3Way(nn.Module):
    def __init__(self, hidden_dim=None, learnable=False):
        super(GatedLinearUnit3Way, self).__init__()
        self.learnable = learnable

        if learnable:
            if hidden_dim is None:
                raise ValueError("You must provide hidden dim for learnable GLU")
            self.hidden_dim = hidden_dim
            self.proj = nn.Linear(hidden_dim, hidden_dim)
            self.mask1 = nn.Linear(hidden_dim, hidden_dim)
            self.mask2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, y, z):
        if self.learnable:
            x = self.proj(x)
            y = self.mask1(y)
            z = self.mask2(z)
        mask1 = torch.sigmoid(y)
        mask2 = torch.sigmoid(z)
        mask = mask1 + mask2
        x = x * mask

        return x


class SubNet(nn.Module):
    """
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    """

    def __init__(self, in_size, hidden_size=32, dropout=0.2):
        """
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        """
        super(SubNet, self).__init__()
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)
        self.out_size = hidden_size

    def forward(self, x, lengths):
        """
        Args:
            x: tensor of shape (batch_size, in_size)
        """
        dropped = self.drop(x)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))

        return y_3


class ModalityProjection(nn.Module):
    def __init__(self, mod1_sz, mod2_sz, proj_sz):
        super(ModalityProjection, self).__init__()
        self.p1 = nn.Linear(mod1_sz, proj_sz)
        self.p2 = nn.Linear(mod2_sz, proj_sz)

    def forward(self, x, y):
        x = self.p1(x)
        y = self.p2(y)

        return x, y


class ModalityProjection3Way(nn.Module):
    def __init__(self, mod1_sz, mod2_sz, mod3_sz, proj_sz):
        super(ModalityProjection3Way, self).__init__()
        self.p1 = nn.Linear(mod1_sz, proj_sz)
        self.p2 = nn.Linear(mod2_sz, proj_sz)
        self.p3 = nn.Linear(mod3_sz, proj_sz)

    def forward(self, x, y, z):
        x = self.p1(x)
        y = self.p2(y)
        z = self.p3(z)

        return x, y, z


class ModalityWeights(nn.Module):
    def __init__(self, mod1_sz, mod2_sz, proj_sz=None, modality_weights=False):
        super(ModalityWeights, self).__init__()
        self.proj, self.mod_w = None, None
        self.proj_sz = mod1_sz if proj_sz is None else proj_sz

        if proj_sz is not None:
            self.proj = ModalityProjection(mod1_sz, mod2_sz, self.proj_sz)

        if modality_weights:
            self.mod_w = nn.Linear(self.proj_sz, 1)

    def forward(self, x, y):
        if self.proj:
            x, y = self.proj(x, y)

        if self.mod_w:
            w = self.mod_w(torch.cat([x.unsqueeze(1), y.unsqueeze(1)], dim=1))
            w = F.softmax(w, dim=1)
            x = x * w[:, 0, ...]
            y = y * w[:, 1, ...]

        return x, y


class ModalityWeights3Way(nn.Module):
    def __init__(self, mod1_sz, mod2_sz, mod3_sz, proj_sz=None, modality_weights=False):
        super(ModalityWeights3Way, self).__init__()
        self.proj, self.mod_w = None, None
        self.proj_sz = mod1_sz if proj_sz is None else proj_sz

        if proj_sz is not None:
            self.proj = ModalityProjection3Way(mod1_sz, mod2_sz, mod3_sz, self.proj_sz)

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


class CommonSpaceFuser(nn.Module):
    def __init__(
        self,
        mod1_sz,
        mod2_sz,
        proj_sz=None,
        modality_weights=False,
        device="cpu",
        extra_args=None,
    ):
        super(CommonSpaceFuser, self).__init__()
        self.mod1_sz = mod1_sz
        self.mod2_sz = mod2_sz
        self.proj_sz = proj_sz if proj_sz is not None else mod1_sz
        self.proj_sz = int(3 * round(float(self.proj_sz) / 3))
        self.device = device
        self.mw = ModalityWeights(
            mod1_sz, mod2_sz, proj_sz=self.proj_sz, modality_weights=modality_weights
        )
        self.w = nn.Parameter(torch.Tensor(2 * self.proj_sz, self.proj_sz))
        self.b = nn.Parameter(torch.Tensor(self.proj_sz))
        self.mask = (
            self.get_mask()

            if extra_args["use_mask"]
            else torch.ones((2 * self.proj_sz, self.proj_sz)).to(device)
        )
        self.out_size = self.proj_sz
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

        if self.b is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.b, -bound, bound)

    def get_mask(self):
        mask = torch.zeros((2 * self.proj_sz, self.proj_sz))
        boundary1 = int(self.proj_sz / 3)
        boundary2 = 2 * int(self.proj_sz / 3)
        mask[: self.proj_sz, :boundary1] = 1
        mask[:, boundary1:boundary2] = 1
        mask[self.proj_sz :, boundary2:] = 0
        mask = mask.to(self.device)

        return mask

    def forward(self, x, y):
        x, y = self.mw(x, y)
        z = torch.cat((x, y), dim=1)
        w_ = self.w * self.mask
        # (B, M1 + M2) x (M1 + M2, P) -> (B, P)
        z = torch.matmul(z, w_) + self.b

        return z


class CatFuser(nn.Module):
    def __init__(
        self,
        mod1_sz,
        mod2_sz,
        proj_sz=None,
        modality_weights=False,
        device="cpu",
        extra_args=None,
    ):
        super(CatFuser, self).__init__()
        self.mw = ModalityWeights(
            mod1_sz, mod2_sz, proj_sz=proj_sz, modality_weights=modality_weights
        )
        self.out_size = mod1_sz + mod2_sz if proj_sz is None else 2 * proj_sz

    def forward(self, x, y):
        x, y = self.mw(x, y)

        return torch.cat([x, y], dim=1)


class CatFuser3Way(nn.Module):
    def __init__(
        self,
        mod1_sz,
        mod2_sz,
        mod3_sz,
        proj_sz=None,
        modality_weights=False,
        device="cpu",
        extra_args=None,
    ):
        super(CatFuser3Way, self).__init__()
        self.mw = ModalityWeights3Way(
            mod1_sz,
            mod2_sz,
            mod3_sz,
            proj_sz=proj_sz,
            modality_weights=modality_weights,
        )
        self.out_size = mod1_sz + mod2_sz + mod3_sz if proj_sz is None else 3 * proj_sz

    def forward(self, x, y, z):
        x, y, z = self.mw(x, y, z)

        return torch.cat([x, y, z], dim=1)


class AddFuser(nn.Module):
    def __init__(
        self,
        mod1_sz,
        mod2_sz,
        proj_sz=None,
        modality_weights=False,
        device="cpu",
        extra_args=None,
    ):
        super(AddFuser, self).__init__()
        self.mw = ModalityWeights(
            mod1_sz, mod2_sz, proj_sz=proj_sz, modality_weights=False
        )
        self.out_size = mod1_sz if proj_sz is None else proj_sz

    def forward(self, x, y):
        x, y = self.mw(x, y)

        return x + y


class AddFuser3Way(nn.Module):
    def __init__(
        self,
        mod1_sz,
        mod2_sz,
        mod3_sz,
        proj_sz=None,
        modality_weights=False,
        device="cpu",
        extra_args=None,
    ):
        super(AddFuser3Way, self).__init__()
        self.mw = ModalityWeights3Way(
            mod1_sz, mod2_sz, mod3_sz, proj_sz=proj_sz, modality_weights=False
        )
        self.out_size = mod1_sz if proj_sz is None else proj_sz

    def forward(self, x, y, z):
        x, y, z = self.mw(x, y, z)

        return x + y + z


class AudioEncoderAverage(nn.Module):
    def __init__(self, cfg, device="cpu"):
        super(AudioEncoderAverage, self).__init__()
        self.audio = SubNet(
            cfg["input_size"], hidden_size=cfg["hidden_size"], dropout=cfg["dropout"]
        )
        self.out_size = self.audio.out_size

        self.bn = None

        if cfg["batchnorm"]:
            self.bn = nn.BatchNorm1d(cfg["input_size"])

    def forward(self, x, lengths):
        x = x.mean(dim=1)

        if self.bn is not None:
            x = self.bn(x.view(-1, x.size(2), x.size(1))).view(-1, x.size(1), x.size(2))
        x = self.audio(x, lengths)

        return x


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


class TextEncoder(nn.Module):
    def __init__(self, cfg, embeddings, vocab_size=None, device="cpu"):
        super(TextEncoder, self).__init__()
        self.text = WordRNN(
            cfg["input_size"],
            embeddings=embeddings,
            vocab_size=vocab_size,
            batch_first=True,
            embeddings_dim=cfg["input_size"],
            embeddings_dropout=0,
            layers=cfg["layers"],
            merge_bi="sum",
            bidirectional=cfg["bidirectional"],
            dropout=cfg["dropout"],
            rnn_type=cfg["rnn_type"],
            packed_sequence=True,
            attention=cfg["attention"],
            device=device,
            finetune_embeddings=False,
            return_hidden=cfg["return_hidden"],
        )
        self.out_size = self.text.out_size

    def forward(self, x, lengths):
        x = self.text(x, lengths)

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
        assert text_cfg["attention"] and audio_cfg["attention"], "Use attention pls."

        self.feedback = feedback

        text_size = fuse_cfg["projection_size"]
        audio_size = fuse_cfg["projection_size"]
        text_cfg["orig_size"] = text_cfg.get("orig_size", text_cfg["input_size"])
        audio_cfg["orig_size"] = audio_cfg.get("orig_size", audio_cfg["input_size"])
        audio_cfg["input_size"] = audio_size
        text_cfg["input_size"] = text_size

        self.audio_projection = nn.Linear(
            audio_cfg["orig_size"], fuse_cfg["projection_size"]
        )

        self.text_projection = None

        if text_cfg["orig_size"] != fuse_cfg["projection_size"]:
            self.text_projection = nn.Linear(
                text_cfg["orig_size"], fuse_cfg["projection_size"]
            )

        self.prefuser = None

        if fuse_cfg["prefuse"]:
            self.prefuser = nn.Linear(
                fuse_cfg["projection_size"], fuse_cfg["projection_size"]
            )

        text_cfg["return_hidden"] = True
        audio_cfg["return_hidden"] = True

        if text_mode == "glove":
            self.text = GloveEncoder(text_cfg, device=device)
        else:
            raise ValueError("Only glove supported for now")

        self.audio = AudioEncoder(audio_cfg, device=device)

        if fuse_cfg["method"] == "cat":
            fuse_cls = CatFuser
        elif fuse_cfg["method"] == "add":
            fuse_cls = AddFuser
        elif fuse_cfg["method"] == "common_space":
            fuse_cls = CommonSpaceFuser
        else:
            raise ValueError('Supported fuse techniques: ["cat", "add"]')

        self.fuser = fuse_cls(
            self.text.out_size,
            self.audio.out_size,
            proj_sz=fuse_cfg["projection_size"],
            modality_weights=fuse_cfg["modality_weights"],
            device=device,
            extra_args=fuse_cfg,
        )
        self.out_size = self.fuser.out_size

    def forward(self, txt, au, lengths):
        au = self.audio_projection(au)

        if self.text_projection is not None:
            txt = self.text_projection(txt)


        if self.prefuser is not None:
            au = self.prefuser(au)
            txt = self.prefuser(txt)

        if self.feedback:
            for _ in range(2):
                txt = self.text(txt, lengths)
                au = self.audio(au, lengths)
                txt = F.glu(torch.cat((txt, au), dim=-1), dim=-1)
                au = F.glu(torch.cat((au, txt), dim=-1), dim=-1)

        txt = self.text(txt, lengths)
        au = self.audio(au, lengths)

        # Sum weighted attention hidden states
        txt = txt.sum(1)
        au = au.sum(1)

        fused = self.fuser(txt, au)

        return fused


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

        audio_size = fuse_cfg["projection_size"]
        text_size = fuse_cfg["projection_size"]
        visual_size = fuse_cfg["projection_size"]
        text_cfg["orig_size"] = text_cfg.get("orig_size", text_cfg["input_size"])
        audio_cfg["orig_size"] = audio_cfg.get("orig_size", audio_cfg["input_size"])
        visual_cfg["orig_size"] = visual_cfg.get("orig_size", visual_cfg["input_size"])
        audio_cfg["input_size"] = audio_size
        text_cfg["input_size"] = text_size
        visual_cfg["input_size"] = visual_size

        self.text_projection = None

        if text_cfg["orig_size"] != fuse_cfg["projection_size"]:
            self.text_projection = nn.Linear(
                text_cfg["orig_size"], fuse_cfg["projection_size"]
            )

        self.audio_projection = nn.Linear(
            audio_cfg["orig_size"], fuse_cfg["projection_size"]
        )

        self.visual_projection = nn.Linear(
            visual_cfg["orig_size"], fuse_cfg["projection_size"]
        )

        self.prefuser = None

        if fuse_cfg["prefuse"]:
            self.prefuser = nn.Linear(
                fuse_cfg["projection_size"], fuse_cfg["projection_size"]
            )

        text_cfg["return_hidden"] = True
        audio_cfg["return_hidden"] = True
        visual_cfg["return_hidden"] = True

        if text_mode == "glove":
            self.text = GloveEncoder(text_cfg, device=device)
        else:
            raise ValueError("Only glove supported for now")

        self.audio = AudioEncoder(audio_cfg, device=device)

        self.visual = VisualEncoder(visual_cfg, device=device)

        if feedback:
            self.glu = GatedLinearUnit3Way()

        if fuse_cfg["method"] == "cat":
            fuse_cls = CatFuser3Way
        elif fuse_cfg["method"] == "add":
            fuse_cls = AddFuser3Way
        elif fuse_cfg["method"] == "common_space":
            raise NotImplementedError
        else:
            raise ValueError('Supported fuse techniques: ["cat", "add"]')

        self.fuser = fuse_cls(
            self.text.out_size,
            self.audio.out_size,
            self.visual.out_size,
            proj_sz=fuse_cfg["projection_size"],
            modality_weights=fuse_cfg["modality_weights"],
            device=device,
            extra_args=fuse_cfg,
        )
        self.out_size = self.fuser.out_size

    def from_pretrained(self, audio_path, visual_path, text_path):
        # load pretrained classifiers
        text_clf = torch.load(text_path)
        audio_clf = torch.load(audio_path)
        visual_clf = torch.load(visual_path)
        # get projections
        self.text_projection = text_clf["proj"]
        self.audio_projection = audio_clf["proj"]
        self.visual_projection = visual_clf["proj"]
        # encoders
        self.text = text_clf["encoder"]
        self.audio = audio_clf["encoder"]
        self.visual = visual_clf["encoder"]

    def forward(self, txt, au, vi, lengths):
        if self.audio_projection is not None:
            au = self.audio_projection(au)

        if self.visual_projection is not None:
            vi = self.visual_projection(vi)

        if self.text_projection is not None:
            txt = self.text_projection(txt)

        if self.prefuser is not None:
            au = self.prefuser(au)
            vi = self.prefuser(vi)
            txt = self.prefuser(txt)

        if self.feedback:
            for _ in range(2):
                txt = self.text(txt, lengths)
                au = self.audio(au, lengths)
                vi = self.visual(vi, lengths)
                txt = self.glu(txt, au, vi)
                au = self.glu(au, txt, vi)
                vi = self.glu(vi, txt, au)

        txt = self.text(txt, lengths)
        au = self.audio(au, lengths)
        vi = self.visual(vi, lengths)

        # Sum weighted attention hidden states
        txt = txt.sum(1)
        au = au.sum(1)
        vi = vi.sum(1)

        fused = self.fuser(txt, au, vi)

        return fused


class AudioVisualTextCoAttentionEncoder(nn.Module):
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
        super(AudioVisualTextCoAttentionEncoder, self).__init__()
        # For now model dim == text dim (the largest). In the future this can be done
        # with individual projection layers for each modality
        assert (
            text_cfg["attention"] and audio_cfg["attention"] and visual_cfg["attention"]
        ), "Use attention pls."

        self.feedback = feedback

        audio_size = fuse_cfg["projection_size"]
        text_size = fuse_cfg["projection_size"]
        visual_size = fuse_cfg["projection_size"]
        text_cfg["orig_size"] = text_cfg.get("orig_size", text_cfg["input_size"])
        audio_cfg["orig_size"] = audio_cfg.get("orig_size", audio_cfg["input_size"])
        visual_cfg["orig_size"] = visual_cfg.get("orig_size", visual_cfg["input_size"])
        audio_cfg["input_size"] = audio_size
        text_cfg["input_size"] = text_size
        visual_cfg["input_size"] = visual_size

        self.text_projection = None

        if text_cfg["orig_size"] != fuse_cfg["projection_size"]:
            self.text_projection = nn.Linear(
                text_cfg["orig_size"], fuse_cfg["projection_size"]
            )

        self.audio_projection = nn.Linear(
            audio_cfg["orig_size"], fuse_cfg["projection_size"]
        )

        self.visual_projection = nn.Linear(
            visual_cfg["orig_size"], fuse_cfg["projection_size"]
        )

        self.prefuser = None

        if fuse_cfg["prefuse"]:
            self.prefuser = nn.Linear(
                fuse_cfg["projection_size"], fuse_cfg["projection_size"]
            )

        assert text_mode == "glove", "Only glove supported for now"

        self.text = CoAttentiveRNN(
            text_cfg["input_size"],
            text_cfg["hidden_size"],
            cross_size=2 * fuse_cfg["projection_size"],
            num_heads=text_cfg["num_heads"],
            batch_first=True,
            layers=text_cfg["layers"],
            merge_bi="sum",
            bidirectional=text_cfg["bidirectional"],
            dropout=text_cfg["dropout"],
            rnn_type=text_cfg["rnn_type"],
            packed_sequence=True,
            attention=text_cfg["attention"],
            device=device,
            return_hidden=True,
        )

        self.audio = CoAttentiveRNN(
            audio_cfg["input_size"],
            audio_cfg["hidden_size"],
            cross_size=2 * fuse_cfg["projection_size"],
            num_heads=text_cfg["num_heads"],
            batch_first=True,
            layers=audio_cfg["layers"],
            merge_bi="sum",
            bidirectional=audio_cfg["bidirectional"],
            dropout=audio_cfg["dropout"],
            rnn_type=audio_cfg["rnn_type"],
            packed_sequence=True,
            attention=audio_cfg["attention"],
            device=device,
            return_hidden=True,
        )

        self.visual = CoAttentiveRNN(
            visual_cfg["input_size"],
            visual_cfg["hidden_size"],
            cross_size=2 * fuse_cfg["projection_size"],
            num_heads=visual_cfg["num_heads"],
            batch_first=True,
            layers=visual_cfg["layers"],
            merge_bi="sum",
            bidirectional=visual_cfg["bidirectional"],
            dropout=visual_cfg["dropout"],
            rnn_type=visual_cfg["rnn_type"],
            packed_sequence=True,
            attention=visual_cfg["attention"],
            device=device,
            return_hidden=True,
        )

        if feedback:
            self.glu = GatedLinearUnit3Way()

        if fuse_cfg["method"] == "cat":
            fuse_cls = CatFuser3Way
        elif fuse_cfg["method"] == "add":
            fuse_cls = AddFuser3Way
        elif fuse_cfg["method"] == "common_space":
            raise NotImplementedError
        else:
            raise ValueError('Supported fuse techniques: ["cat", "add"]')

        self.fuser = fuse_cls(
            self.text.out_size,
            self.audio.out_size,
            self.visual.out_size,
            proj_sz=fuse_cfg["projection_size"],
            modality_weights=fuse_cfg["modality_weights"],
            device=device,
            extra_args=fuse_cfg,
        )
        self.out_size = self.fuser.out_size

    def from_pretrained(self, audio_path, visual_path, text_path):
        text_model = torch.load(text_path)
        text_encoder = text_model["encoder"]
        self.text_encoder = text_encoder
        raise NotImplementedError


    def forward(self, txt, au, vi, lengths):
        au = self.audio_projection(au)
        vi = self.visual_projection(vi)

        if self.text_projection is not None:
            txt = self.text_projection(txt)

        if self.prefuser is not None:
            au = self.prefuser(au)
            vi = self.prefuser(vi)
            txt = self.prefuser(txt)

        if self.feedback:
            for _ in range(2):
                txt1 = self.text(txt, torch.cat([au, vi], dim=-1), lengths)
                au1 = self.audio(au, torch.cat([txt, vi], dim=-1), lengths)
                vi1 = self.visual(vi, torch.cat([txt, au], dim=-1), lengths)
                txt = txt1
                au = au1
                vi = vi1
                txt = self.glu(txt, au, vi)
                au = self.glu(au, txt, vi)
                vi = self.glu(vi, txt, au)

        txt = self.text(txt, torch.cat([au, vi], dim=-1), lengths)
        au = self.audio(au, torch.cat([txt, vi], dim=-1), lengths)
        vi = self.visual(vi, torch.cat([txt, au], dim=-1), lengths)

        # Sum weighted attention hidden states
        txt = txt.sum(1)
        au = au.sum(1)
        vi = vi.sum(1)

        fused = self.fuser(txt, au, vi)

        return fused



# class FeedbackAudioTextEncoder(nn.Module):
#     def __init__(
#         self,
#         embeddings=None,
#         vocab_size=None,
#         audio_cfg=None,
#         text_cfg=None,
#         fuse_cfg=None,
#         text_mode="glove",
#         device="cpu",
#     ):
#         super(FeedbackAudioTextEncoder, self).__init__()
#         assert fuse_cfg["projection_size"] == text_cfg["input_size"]

#         audio_size = fuse_cfg["projection_size"]
#         text_cfg["orig_size"] = text_cfg.get("orig_size", text_cfg["input_size"])
#         audio_cfg["orig_size"] = audio_cfg.get("orig_size", audio_cfg["input_size"])
#         audio_cfg["input_size"] = audio_size

#         self.audio_projection = nn.Linear(
#             audio_cfg["orig_size"], fuse_cfg["projection_size"]
#         )

#         self.prefuser = None

#         if fuse_cfg["prefuse"]:
#             self.prefuser = nn.Linear(
#                 fuse_cfg["projection_size"], fuse_cfg["projection_size"]
#             )

#         text_cfg["return_hidden"] = True
#         audio_cfg["return_hidden"] = True

#         if text_mode == "glove":
#             self.text = GloveEncoder(text_cfg, device=device)
#         else:
#             raise ValueError("Only glove input supported")

#         self.audio = AudioEncoder(audio_cfg, device=device)

#         if fuse_cfg["method"] == "cat":
#             fuse_cls = CatFuser
#         elif fuse_cfg["method"] == "add":
#             fuse_cls = AddFuser
#         elif fuse_cfg["method"] == "common_space":
#             fuse_cls = CommonSpaceFuser
#         else:
#             raise ValueError('Supported fuse techniques: ["cat", "add"]')

#         self.fuser = fuse_cls(
#             self.text.out_size,
#             self.audio.out_size,
#             proj_sz=fuse_cfg["projection_size"],
#             modality_weights=fuse_cfg["modality_weights"],
#             device=device,
#             extra_args=fuse_cfg,
#         )
#         self.out_size = self.fuser.out_size

#     def forward(self, txt, au, lengths):
#         au = self.audio_projection(au)

#         if self.prefuser is not None:
#             au = self.prefuser(au)
#             txt = self.prefuser(txt)

#         for _ in range(2):
#             txt = self.text(txt, lengths)
#             au = self.audio(au, lengths)
#             txt = F.glu(torch.cat((txt, au), dim=-1), dim=-1)
#             au = F.glu(torch.cat((au, txt), dim=-1), dim=-1)
#         txt = self.text(txt, lengths)
#         au = self.audio(au, lengths)
#         txt = txt.sum(1)
#         au = au.sum(1)
#         fused = self.fuser(txt, au)

#         return fused


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

        # enc_cls = AudioTextEncoder if not feedback else FeedbackAudioTextEncoder
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

    def forward(self, inputs):
        out = self.encoder(inputs["text"], inputs["audio"], inputs["lengths"])

        return self.classifier(out)


class GloveClassifier(nn.Module):
    def __init__(self, cfg, num_classes=1, projection_size=300, device="cpu"):
        super(GloveClassifier, self).__init__()
        self.proj = None
        if cfg["hidden_size"] != projection_size:
            self.proj = nn.Linear(cfg["input_size"], projection_size)
        self.encoder = AudioEncoder(cfg, device=device)
        self.classifier = nn.Linear(self.encoder.out_size, num_classes)

    def forward(self, x, lengths):
        if self.proj is not None:
            x = self.proj(x)
        x = self.encoder(x, lengths)
        return self.classifier(x)

class AudioClassifier(nn.Module):
    def __init__(self, cfg, num_classes=1, projection_size=300, device="cpu"):
        super(AudioClassifier, self).__init__()
        self.proj = None
        if cfg["hidden_size"] != projection_size:
            self.proj = nn.Linear(cfg["input_size"], projection_size)
        self.encoder = AudioEncoder(cfg, device=device)
        self.classifier = nn.Linear(self.encoder.out_size, num_classes)

    def forward(self, x, lengths):
        if self.proj is not None:
            x = self.proj(x)
        x = self.encoder(x, lengths)
        return self.classifier(x)

class VisualClassifier(nn.Module):
    def __init__(self, cfg, num_classes=1, projection_size=300, device="cpu"):
        super(VisualClassifier, self).__init__()
        self.proj = None
        if cfg["hidden_size"] != projection_size:
            self.proj = nn.Linear(cfg["input_size"], projection_size)
        self.encoder = VisualEncoder(cfg, device=device)
        self.classifier = nn.Linear(self.encoder.out_size, num_classes)

    def forward(self, x, lengths):
        if self.proj is not None:
            x = self.proj(x)
        x = self.encoder(x, lengths)
        return self.classifier(x)

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


class AudioVisualTextCoAttentionClassifier(nn.Module):
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
        super(AudioVisualTextCoAttentionClassifier, self).__init__()
        self.modalities = modalities

        assert "text" in modalities or "glove" in modalities, "No text"
        assert "audio" in modalities, "No audio"
        assert "visual" in modalities, "No visual"

        self.encoder = AudioVisualTextCoAttentionEncoder(
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
