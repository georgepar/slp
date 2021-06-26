from typing import List, Optional, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from slp.modules.attention import Attention
from slp.modules.rnn import AttentiveRNN
from slp.modules.symattention import SymmetricAttention


def _all_equal(sizes):
    return all(s == sizes[0] for s in sizes)


class Conv1dProj(nn.Module):
    """Project features for N modalities using 1D convolutions

    Args:
        modality_sizes (List[int]): List of number of features for each modality. E.g. for MOSEI:
            [300, 74, 35]
        projection_size (int): Output number of features for each modality
        kernel_size (int): Convolution kernel size
        padding (int): Convlution amount of padding
        bias (bool): Use bias in convolutional layers
    """

    def __init__(
        self,
        modality_sizes: List[int],
        projection_size: int,
        kernel_size: int = 1,
        padding: int = 0,
        bias: bool = False,
    ):
        super(Conv1dProj, self).__init__()
        self.p = [
            nn.Conv1d(
                sz, projection_size, kernel_size=kernel_size, padding=padding, bias=bias
            )
            for sz in modality_sizes
        ]

    def forward(self, *mods):
        mods = [
            self.p[i](m.transpose(1, 2)).transpose(1, 2) for i, m in enumerate(mods)
        ]

        return mods


class LinearProj(nn.Module):
    """Project features for N modalities using feedforward layers

    Args:
        modality_sizes (List[int]): List of number of features for each modality. E.g. for MOSEI:
            [300, 74, 35]
        bias (bool): Use bias in feedforward layers
    """

    def __init__(
        self, modality_sizes: List[int], projection_size: int, bias: bool = True
    ):
        super(LinearProj, self).__init__()
        self.p = [nn.Linear(sz, projection_size, bias=bias) for sz in modality_sizes]

    def forward(self, *mods):
        mods = [self.p[i](m) for i, m in enumerate(mods)]

        return mods


class ModalityProjection(nn.Module):
    """Adapter module to project features for N modalities using 1D convolutions or feedforward

    Args:
        modality_sizes (List[int]): List of number of features for each modality. E.g. for MOSEI:
            [300, 74, 35]
        projection_size (int): Output number of features for each modality
        kernel_size (int): Convolution kernel size. Used when mode=="conv"
        padding (int): Convlution amount of padding. Used when mode=="conv"
        bias (bool): Use bias
        mode (Optional[str]): Projection method.
            linear -> LinearProj
            conv|conv1d|convolutional -> Conv1dProj
    """

    def __init__(
        self,
        modality_sizes: List[int],
        projection_size: int,
        kernel_size: int = 1,
        padding: int = 0,
        bias: bool = True,
        mode: Optional[str] = None,
    ):
        super(ModalityProjection, self).__init__()

        if mode is None:
            self.p: Optional[Union[LinearProj, Conv1dProj]] = None
        elif mode == "linear":
            self.p = LinearProj(modality_sizes, projection_size, bias=bias)
        elif mode == "conv" or mode == "conv1d" or mode == "convolutional":
            self.p = Conv1dProj(
                modality_sizes,
                projection_size,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            )
        else:
            raise ValueError(
                "Supported mode=[linear|conv|conv1d|convolutional]."
                "conv, conv1d and convolutional are equivalent."
            )

    def forward(self, *mods):
        if self.p is None:
            return mods
        mods = self.p(*mods)

        return mods


class ModalityWeights(nn.Module):
    """Multiply each modality features with a learnable weight

    learnable_weight = softmax(Linear(modality_features))

    Args:
        feature_size (int): All modalities are assumed to be projected into a space with the same
            number of features.

    """

    def __init__(self, feature_size: int):
        super(ModalityWeights, self).__init__()

        self.mod_w = nn.Linear(feature_size, 1)

    def forward(self, *mods):
        weight = self.mod_w(torch.cat([x.unsqueeze(1) for x in mods], dim=1))
        weight = F.softmax(weight, dim=1)
        mods = [m * weight[:, i, ...] for i, m in enumerate(mods)]

        return mods


class TimestepAggregator(nn.Module):
    """Aggregate features from all timesteps into a single representation.

    Three methods supported:
        sum: Sum features from all timesteps
        mean: Average features from all timesteps
        rnn: Use the output from an attentive RNN

    Args:
        feature_size (int): The number of features for the input fused representations
        hidden_size (Optional[int]): The hidden size of the rnn. Used only when mode == rnn
        batch_first (bool): Input tensors are in batch first configuration. Leave this as true
            except if you know what you are doing
        bidirectional (bool): Use a bidirectional rnn. Used only when mode == rnn
        merge_bi (str): Method to fuse the bidirectional hidden states. Used only when mode == rnn
        attention (bool): Use an attention mechanism in the rnn output. Used only when mode == rnn
        mode (str): The timestep aggregation method
            sum: Sum hidden states
            mean: Average hidden states
            rnn: Use the output of an Attentive RNN
    """

    def __init__(
        self,
        feature_size: int,
        hidden_size: Optional[int] = None,
        batch_first: bool = True,
        bidirectional: bool = True,
        merge_bi: str = "cat",
        attention: bool = True,
        mode: str = "sum",
    ):
        super(TimestepAggregator, self).__init__()
        assert mode in ["sum", "mean", "rnn"], "Unsupported timestep aggregation"
        self.mode = mode
        self.feature_size = feature_size
        self.hidden_size = hidden_size if hidden_size is not None else feature_size
        self.aggregation_dim = 0 if not batch_first else 1

        if mode == "rnn":
            self.rnn = AttentiveRNN(
                feature_size,
                hidden_size=self.hidden_size,
                batch_first=batch_first,
                bidirectional=bidirectional,
                merge_bi=merge_bi,
                attention=attention,
                return_hidden=False,  # We want to aggregate all hidden states.
            )

    @property
    def out_size(self) -> int:
        if self.mode == "sum" or self.mode == "mean":
            return self.feature_size
        elif self.mode == "rnn":
            return self.rnn.out_size
        else:
            raise ValueError("Unsupported timestep aggregation method")

    def forward(self, x, lengths=None):
        if x.ndim == 2:
            return x

        if x.ndim != 3:
            raise ValueError("Expected 3 dimensional tensor [B, L, D] or [L, B, D]")

        if self.mode == "sum":
            return x.sum(self.aggregation_dim)
        elif self.mode == "mean":
            return x.mean(self.aggregation_dim)
        elif self.mode == "rnn":
            return self.rnn(x, lengths)
        else:
            raise ValueError("Unsupported timestep aggregation method")


class BaseFuser(nn.Module):
    """Base fuser class.

    Our fusion methods are separated in direct and combinatorial.
    An example for direct fusion is concatenation, where feature vectors of N modalities
    are concatenated into a fused vector.
    When performing combinatorial fusion all crossmodal relations are examined (e.g. text -> audio,
    text -> visual, audio -> visaul etc.)
    In the current implementation, combinatorial fusion is implemented for 3 input modalities

    Args:
        feature_size (int): Assume all modality representations have the same feature_size
        n_modalities (int): Number of input modalities
        **extra_kwargs (dict): Extra keyword arguments to maintain interoperability of children
            classes
    """

    def __init__(
        self,
        feature_size: int,
        n_modalities: int,
        **extra_kwargs,
    ):
        super(BaseFuser, self).__init__()
        self.feature_size = feature_size
        self.n_modalities = n_modalities

    def _check_3_modalities(self, aux_log="This fuser"):
        """Check if this fuser is created for 3 modalities

        Some fusers are implemented particularly for 3 modalities (txt, audio, visual).
        Check at instantiation time if they are misused

        Args:
            aux_log (str): Auxiliary log to prepend to error log

        """

        if self.n_modalities != 3:
            raise ValueError(
                f"{aux_log} implemented for 3 modalities [text, audio, visual]"
            )

    @property
    def out_size(self) -> int:
        """Output feature size.

        Each fuser specifies its output feature size
        """
        raise NotImplementedError

    def fuse(self, *mods, lengths=None):
        """Abstract method to fuse the modality representations

        Children classes should implement this method

        Args:
            *mods (vararg): List of modality tensors
            lengths (Optional[Tensor]): Lengths of each modality

        """
        raise NotImplementedError

    def forward(self, *mods, lengths=None):
        fused = self.fuse(*mods, lengths=lengths)

        return fused


class CatFuser(BaseFuser):
    """Fuse by concatenating modality representations

    o = m1 || m2 || m3 ...

    Args:
        feature_size (int): Assume all modality representations have the same feature_size
        n_modalities (int): Number of input modalities
        **extra_kwargs (dict): Extra keyword arguments to maintain interoperability of children
            classes
    """

    def __init__(self, feature_size, n_modalities, **extra_kwargs):
        super(CatFuser, self).__init__(feature_size, n_modalities, **extra_kwargs)

    @property
    def out_size(self) -> int:
        return self.n_modalities * self.feature_size

    def fuse(self, *mods, lengths=None):
        return torch.cat(mods, dim=1)


class AddFuser(BaseFuser):
    """Fuse by adding modality representations

    o = m1 + m2 + m3 ...

    Args:
        feature_size (int): Assume all modality representations have the same feature_size
        n_modalities (int): Number of input modalities
        **extra_kwargs (dict): Extra keyword arguments to maintain interoperability of children
            classes
    """

    def __init__(self, feature_size, n_modalities, **kwargs):
        super(AddFuser, self).__init__(feature_size, n_modalities, **kwargs)

    @property
    def out_size(self) -> int:
        return self.feature_size

    def fuse(self, *mods, lengths=None):
        out = torch.zeros_like(mods[0])

        for m in mods:
            out = out + m

        return out


class BilinearFuser(BaseFuser):
    """Bilinear combinatorial fusion

    Obtain all crossmodal relations with bilinear transformation

    Args:
        feature_size (int): Assume all modality representations have the same feature_size
        n_modalities (int): Number of input modalities
        use_all_trimodal (bool): Represent all trimodal interactions (t -> av, a -> tv, v -> ta).
            If False only t -> av is used. Default value is False.
        **extra_kwargs (dict): Extra keyword arguments to maintain interoperability of children
            classes
    """

    def __init__(self, feature_size, n_modalities, use_all_trimodal=False, **kwargs):
        super(BilinearFuser, self).__init__(feature_size, n_modalities, **kwargs)
        self._check_3_modalities(aux_log="BilinearFuser")
        sz = feature_size
        self.use_all_trimodal = use_all_trimodal
        self.ta = nn.Bilinear(sz, sz, sz)
        self.at = nn.Bilinear(sz, sz, sz)
        self.va = nn.Bilinear(sz, sz, sz)
        self.av = nn.Bilinear(sz, sz, sz)
        self.tv = nn.Bilinear(sz, sz, sz)
        self.vt = nn.Bilinear(sz, sz, sz)
        self.tav = nn.Bilinear(sz, sz, sz)

        if use_all_trimodal:
            self.vat = nn.Bilinear(sz, sz, sz)
            self.atv = nn.Bilinear(sz, sz, sz)

    @property
    def out_size(self) -> int:
        if self.use_all_trimodal:
            return 9 * self.feature_size
        else:
            return 7 * self.feature_size

    def fuse(self, *mods, lengths=None):
        txt, au, vi = mods
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

        if self.use_all_trimodal:
            vat = self.vat(vi, ta)
            atv = self.atv(au, tv)

        if not self.use_all_trimodal:
            # B x L x 7*D
            fused = torch.cat([txt, au, vi, ta, tv, av, tav], dim=-1)
        else:
            # B x L x 9*D
            fused = torch.cat([txt, au, vi, ta, tv, av, tav, vat, atv], dim=-1)

        return fused


class AttentionFuser(BaseFuser):
    """Attention combinatorial fusion

    Obtain all crossmodal relations using attention mechanisms

    Args:
        feature_size (int): Assume all modality representations have the same feature_size
        n_modalities (int): Number of input modalities
        residual (bool): Use Vilbert-like residual connection in SymmetricAttention mechanisms
        use_all_trimodal (bool): Represent all trimodal interactions (t -> av, a -> tv, v -> ta).
            If False only t -> av is used. Default value is False.
        **extra_kwargs (dict): Extra keyword arguments to maintain interoperability of children
            classes
    """

    def __init__(
        self,
        feature_size,
        n_modalities,
        residual=True,
        use_all_trimodal=False,
        **kwargs,
    ):
        super(AttentionFuser, self).__init__(feature_size, n_modalities, **kwargs)
        self._check_3_modalities(aux_log="AttentionFuser")
        sz = feature_size
        self.ta = SymmetricAttention(
            attention_size=sz,
            dropout=0.1,
            residual=residual,
            layernorm=False,
        )

        self.va = SymmetricAttention(
            attention_size=sz,
            dropout=0.1,
            residual=residual,
            layernorm=False,
        )

        self.tv = SymmetricAttention(
            attention_size=sz,
            dropout=0.1,
            residual=residual,
            layernorm=False,
        )

        self.tav = Attention(
            attention_size=sz,
            dropout=0.1,
        )

        self.use_all_trimodal = use_all_trimodal

        if use_all_trimodal:

            self.vat = Attention(
                attention_size=sz,
                dropout=0.1,
            )

            self.atv = Attention(
                attention_size=sz,
                dropout=0.1,
            )

    @property
    def out_size(self) -> int:
        if self.use_all_trimodal:
            return 9 * self.feature_size
        else:
            return 7 * self.feature_size

    def fuse(self, *mods, lengths=None):
        txt, au, vi = mods
        ta, at = self.ta(txt, au)
        va, av = self.va(vi, au)
        tv, vt = self.tv(txt, vi)

        av = va + av
        tv = vt + tv
        ta = ta + at

        tav, _ = self.tav(txt, queries=av)

        if self.use_all_trimodal:
            vat, _ = self.vat(vi, queries=ta)
            atv, _ = self.atv(au, queries=tv)

        if not self.use_all_trimodal:
            # B x L x 7*D
            fused = torch.cat([txt, au, vi, ta, tv, av, tav], dim=-1)
        else:
            # B x L x 9*D
            fused = torch.cat([txt, au, vi, ta, tv, av, tav, vat, atv], dim=-1)

        return fused


SUPPORTED_FUSERS = {
    "cat": CatFuser,
    "add": AddFuser,
    "sum": AddFuser,
    "bilinear": BilinearFuser,
    "attention": AttentionFuser,
    "attn": AttentionFuser,
}


def make_fuser(fusion_method, feature_size, n_modalities, **kwargs):
    if fusion_method not in SUPPORTED_FUSERS.keys():
        raise NotImplementedError(f"The supported fusers are {SUPPORTED_FUSERS.keys()}")

    return SUPPORTED_FUSERS[fusion_method](feature_size, n_modalities, **kwargs)


class BaseFusionPipeline(nn.Module):
    """Base class for a fusion pipeline

    Inherit this class to implement a fusion pipeline

    """

    def __init__(self, *args, **kwargs):
        super(BaseFusionPipeline, self).__init__()

    @property
    def out_size(self) -> int:
        raise NotImplementedError


class FuseAggregateTimesteps(BaseFusionPipeline):
    """Fuse input feature sequences and aggregate across timesteps

    Fuser -> TimestepAggregator

    Args:
        feature_size (int): The input modality representations dimension
        n_modalities (int): Number of input modalities
        output_size (Optional[int]): Required output size. If not provided,
            output_size = fuser.out_size
        fusion_method (str): Select which fuser to use [cat|sum|attention|bilinear]
        timestep_aggregation_method (str): TimestepAggregator method [cat|sum|rnn]
        batch_first (bool): Input tensors are in batch first configuration. Leave this as true
            except if you know what you are doing
        **fuser_kwargs (dict): Extra keyword arguments to instantiate fuser
    """

    def __init__(
        self,
        feature_size: int,
        n_modalities: int,
        output_size: Optional[int] = None,
        fusion_method: str = "cat",
        timestep_aggregation_method: str = "sum",
        batch_first: bool = True,
        **fuser_kwargs,
    ):
        super(FuseAggregateTimesteps, self).__init__(
            feature_size, n_modalities, fusion_method=fusion_method
        )
        self.fuser = make_fuser(
            fusion_method, feature_size, n_modalities, **fuser_kwargs
        )
        output_size = (  # bidirectional rnn. fused_size / 2 results to fused_size outputs
            output_size if output_size is not None else self.fuser.out_size // 2
        )
        self.timestep_aggregator = TimestepAggregator(
            self.fuser.out_size,
            hidden_size=output_size,
            mode=timestep_aggregation_method,
            batch_first=batch_first,
        )

    @property
    def out_size(self) -> int:
        return cast(int, self.timestep_aggregator.out_size)

    def forward(self, *mods, lengths=None):
        fused = self.fuser(*mods, lengths=lengths)
        out = self.timestep_aggregator(fused, lengths=lengths)

        return out


class ProjectFuseAggregate(nn.Module):
    """Project input feature sequences, fuse and aggregate across timesteps

    ModalityProjection -> Optional[ModalityWeights] -> Fuser -> TimestepAggregator

    Args:
        modality_sizes (List[int]): List of input modality representations dimensions
        projection_size (int): Project all modalities to have this feature size
        projection_type (str): Projection method [linear|conv]
        fusion_method (str): Select which fuser to use [cat|sum|attention|bilinear]
        timestep_aggregation_method (str): TimestepAggregator method [cat|sum|rnn]
        modality_weights (bool): Multiply projected modality representations with learnable
            weights. Default value is False.
        batch_first (bool): Input tensors are in batch first configuration. Leave this as true
            except if you know what you are doing
        **fuser_kwargs (dict): Extra keyword arguments to instantiate fuser
    """

    def __init__(
        self,
        modality_sizes: List[int],
        projection_size: int,
        projection_type: str,
        fusion_method="cat",
        timestep_aggregation_method="sum",
        modality_weights: bool = False,
        batch_first: bool = True,
        **fuser_kwargs,
    ):
        super(ProjectFuseAggregate, self).__init__()
        n_modalities = len(modality_sizes)
        self.projection = ModalityProjection(
            modality_sizes, projection_size, mode=projection_type
        )
        self.modality_weights = None

        if modality_weights:
            self.modality_weights = ModalityWeights(projection_size)

        self.fuser = make_fuser(
            fuser_kwargs, projection_size, n_modalities, **fuser_kwargs
        )

        self.timestep_aggregator = TimestepAggregator(
            self.fuser.out_size,
            hidden_size=projection_size,
            batch_first=batch_first,
            mode=timestep_aggregation_method,
        )

    @property
    def out_size(self) -> int:
        return self.timestep_aggregator.out_size

    def forward(self, *mods, lengths=None):
        mods = self.projection(*mods)
        mods = self.modality_weights(*mods)
        fused = self.fuser(*mods, lengths=lengths)
        fused = self.timestep_aggregator(fused, lengths=lengths)

        return fused
