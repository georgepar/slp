from abc import ABCMeta, abstractmethod
from typing import List, Optional, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from slp.modules.attention import Attention
from slp.modules.rnn import AttentiveRNN
from slp.modules.symattention import SymmetricAttention


class Conv1dProjection(nn.Module):
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
        super(Conv1dProjection, self).__init__()
        self.p = nn.ModuleList(
            [
                nn.Conv1d(
                    sz,
                    projection_size,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=bias,
                )
                for sz in modality_sizes
            ]
        )

    def forward(self, *mods: torch.Tensor) -> List[torch.Tensor]:
        """Project modality representations to a given number of features using Conv1d layers
        Example:
            # Inputs:
            #    text: (B, L, 300)
            #    audio: (B, L, 74)
            #    visual: (B, L, 35)
            # Outputs:
            #    text_p: (B, L, 100)
            #    audio_p: (B, L, 100)
            #    visual_p: (B, L, 100)
            c_proj = Conv1dProjection([300, 74, 35], 100)
            text_p, audio_p, visual_p = c_proj(text, audio, visual)

        Args:
            *mods: Variable length tensors list

        Returns:
            List[torch.Tensor]: Variable length projected tensors list
        """
        mods_o: List[torch.Tensor] = [
            self.p[i](m.transpose(1, 2)).transpose(1, 2) for i, m in enumerate(mods)
        ]

        return mods_o


class LinearProjection(nn.Module):
    """Project features for N modalities using feedforward layers

    Args:
        modality_sizes (List[int]): List of number of features for each modality. E.g. for MOSEI:
            [300, 74, 35]
        bias (bool): Use bias in feedforward layers
    """

    def __init__(
        self, modality_sizes: List[int], projection_size: int, bias: bool = True
    ):
        super(LinearProjection, self).__init__()
        self.p = nn.ModuleList(
            [nn.Linear(sz, projection_size, bias=bias) for sz in modality_sizes]
        )

    def forward(self, *mods: torch.Tensor) -> List[torch.Tensor]:
        """Project modality representations to a given number of features using Linear layers
        Example:
            # Inputs:
            #    text: (B, L, 300)
            #    audio: (B, L, 74)
            #    visual: (B, L, 35)
            # Outputs:
            #    text_p: (B, L, 100)
            #    audio_p: (B, L, 100)
            #    visual_p: (B, L, 100)
            l_proj = LinearProjection([300, 74, 35], 100)
            text_p, audio_p, visual_p = l_proj(text, audio, visual)

        Args:
            *mods: Variable length tensor list

        Returns:
            List[torch.Tensor]: Variable length projected tensors list
        """
        mods_o: List[torch.Tensor] = [self.p[i](m) for i, m in enumerate(mods)]

        return mods_o


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
            linear -> LinearProjection
            conv|conv1d|convolutional -> Conv1dProjection
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
            self.p: Optional[Union[LinearProjection, Conv1dProjection]] = None
        elif mode == "linear":
            self.p = LinearProjection(modality_sizes, projection_size, bias=bias)
        elif mode == "conv" or mode == "conv1d" or mode == "convolutional":
            self.p = Conv1dProjection(
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

    def forward(self, *mods: torch.Tensor) -> List[torch.Tensor]:
        """Project modality representations to a given number of features
        Example:
            # Inputs:
            #    text: (B, L, 300)
            #    audio: (B, L, 74)
            #    visual: (B, L, 35)
            # Outputs:
            #    text_p: (B, L, 100)
            #    audio_p: (B, L, 100)
            #    visual_p: (B, L, 100)
            l_proj = ModalityProjection([300, 74, 35], 100, mode="linear")
            text_p, audio_p, visual_p = l_proj(text, audio, visual)

        Example:
            # Inputs:
            #    text: (B, L, 300)
            #    audio: (B, L, 74)
            #    visual: (B, L, 35)
            # Outputs:
            #    text_p: (B, L, 300)
            #    audio_p: (B, L, 74)
            #    visual_p: (B, L, 35)
            l_proj = ModalityProjection([300, 74, 35], 100, mode=None)
            text_p, audio_p, visual_p = l_proj(text, audio, visual)


        Args:
            *mods: Variable length tensor list

        Returns:
            List[torch.Tensor]: Variable length projected tensors list
        """

        if self.p is None:
            return list(mods)
        mods_o: List[torch.Tensor] = self.p(*mods)

        return mods_o


class ModalityWeights(nn.Module):
    """Multiply each modality features with a learnable weight

    i: modality index
    learnable_weight[i] = softmax(Linear(modality_features[i]))
    output_modality[i] = learnable_weight * modality_features[i]

    Args:
        feature_size (int): All modalities are assumed to be projected into a space with the same
            number of features.

    """

    def __init__(self, feature_size: int):
        super(ModalityWeights, self).__init__()

        self.mod_w = nn.Linear(feature_size, 1)

    def forward(self, *mods: torch.Tensor) -> List[torch.Tensor]:
        """Use learnable weights to multiply modality features

        Example:
            # Inputs:
            #    text: (B, L, 100)
            #    audio: (B, L, 100)
            #    visual: (B, L, 100)
            # Outputs:
            #    text_p: (B, L, 100)
            #    audio_p: (B, L, 100)
            #    visual_p: (B, L, 100)
            mw = ModalityWeights(100)
            text_w, audio_w, visual_w = mw(text, audio, visual)

        The operation is summarized as:

        w_x = softmax(W * x + b)
        w_y = softmax(W * y + b)
        x_out = w_x * x
        y_out = w_y * y

        Args:
            *mods: Variable length tensor list

        Returns:
            List[torch.Tensor]: Variable length reweighted tensors list
        """
        weight = self.mod_w(torch.cat([x.unsqueeze(1) for x in mods], dim=1))
        weight = F.softmax(weight, dim=1)
        mods_o: List[torch.Tensor] = [m * weight[:, i, ...] for i, m in enumerate(mods)]

        return mods_o


class BaseTimestepsPooler(nn.Module, metaclass=ABCMeta):
    """Abstract base class for Timesteps Poolers

    Timesteps Poolers aggregate the features for different timesteps

    Given a tensor with dimensions [BatchSize, Length, Dim]
    they return an aggregated tensor with dimensions [BatchSize, Dim]


    Args:
        feature_size (int): Feature dimension
        batch_first (bool): Input tensors are in batch first configuration. Leave this as true
            except if you know what you are doing
        **kwargs: Variable keyword arguments for subclasses
    """

    def __init__(self, feature_size: int, batch_first: bool = True, **kwargs):
        super(BaseTimestepsPooler, self).__init__()
        self.pooling_dim = 0 if not batch_first else 1
        self.feature_size = feature_size

    @property
    def out_size(self) -> int:
        """Define the feature size of the returned tensor

        Returns:
            int: The feature dimension of the output tensor
        """

        return self.feature_size

    @abstractmethod
    def _pool(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Pool features of input tensor across timesteps

        Abstract method to be implemented by subclasses

        Args:
            x (torch.Tensor): [B, L, D] Input sequence
            lengths (Optional[torch.Tensor]): Optional unpadded sequence lengths for input tensor

        Returns:
            torch.Tensor: [B, D] Output aggregated features across timesteps
        """
        pass

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Pool features of input tensor across timesteps

        Args:
            x (torch.Tensor): [B, L, D] Input sequence
            lengths (Optional[torch.Tensor]): Optional unpadded sequence lengths for input tensor

        Returns:
            torch.Tensor: [B, D] Output aggregated features across timesteps
        """

        if x.ndim == 2:
            return x

        if x.ndim != 3:
            raise ValueError("Expected 3 dimensional tensor [B, L, D] or [L, B, D]")

        return self._pool(x, lengths=lengths)


class SumPooler(BaseTimestepsPooler):
    def _pool(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Sum features of input tensor across timesteps

        Args:
            x (torch.Tensor): [B, L, D] Input sequence
            lengths (Optional[torch.Tensor]): Optional unpadded sequence lengths for input tensor

        Returns:
            torch.Tensor: [B, D] Output aggregated features across timesteps
        """

        return x.sum(self.pooling_dim)


class MeanPooler(BaseTimestepsPooler):
    def _pool(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Average features of input tensor across timesteps

        Args:
            x (torch.Tensor): [B, L, D] Input sequence
            lengths (Optional[torch.Tensor]): Optional unpadded sequence lengths for input tensor

        Returns:
            torch.Tensor: [B, D] Output aggregated features across timesteps
        """

        return x.mean(self.pooling_dim)


class MaxPooler(BaseTimestepsPooler):
    def _pool(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Max pooling of features of input tensor across timesteps

        Args:
            x (torch.Tensor): [B, L, D] Input sequence
            lengths (Optional[torch.Tensor]): Optional unpadded sequence lengths for input tensor

        Returns:
            torch.Tensor: [B, D] Output aggregated features across timesteps
        """
        x, _ = x.max(self.pooling_dim)

        return x


class RnnPooler(BaseTimestepsPooler):
    """Aggregate features of the input tensor using an AttentiveRNN

    Args:
        feature_size (int): Feature dimension
        hidden_size (int): Hidden dimension
        batch_first (bool): Input tensors are in batch first configuration. Leave this as true
            except if you know what you are doing
        bidirectional (bool): Use bidirectional RNN. Defaults to True
        merge_bi (str): How bidirectional states are merged. Defaults to "cat"
        attention (bool): Use attention for the RNN output states
        **kwargs: Variable keyword arguments
    """

    def __init__(
        self,
        feature_size: int,
        hidden_size: Optional[int] = None,
        batch_first: bool = True,
        bidirectional: bool = True,
        merge_bi: str = "cat",
        attention: bool = True,
        **kwargs,
    ):
        super(RnnPooler, self).__init__(feature_size, batch_first=batch_first, **kwargs)
        self.hidden_size = hidden_size if hidden_size is not None else feature_size
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
        """Define the feature size of the returned tensor

        Returns:
            int: The feature dimension of the output tensor
        """

        return self.rnn.out_size

    def _pool(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Pass input sequence through an AttentiveRNN and return the weighted average of hidden
        states.

        Args:
            x (torch.Tensor): [B, L, D] Input sequence
            lengths (Optional[torch.Tensor]): Optional unpadded sequence lengths for input tensor

        Returns:
            torch.Tensor: [B, D] Output aggregated features across timesteps
        """

        out: torch.Tensor = self.rnn(x, lengths)

        return out


SUPPORTED_POOLERS = {
    "sum": SumPooler,
    "mean": MeanPooler,
    "max": MaxPooler,
    "rnn": RnnPooler,
}
"""Supported poolers
"""


class TimestepsPooler(BaseTimestepsPooler):
    """Aggregate features from all timesteps into a single representation.

    Three methods supported:
        sum: Sum features from all timesteps
        mean: Average features from all timesteps
        rnn: Use the output from an attentive RNN

    Args:
        feature_size (int): The number of features for the input fused representations
        batch_first (bool): Input tensors are in batch first configuration. Leave this as true
            except if you know what you are doing
        mode (str): The timestep pooling method
            sum: Sum hidden states
            mean: Average hidden states
            rnn: Use the output of an Attentive RNN
    """

    def __init__(
        self, feature_size: int, mode: str = "sum", batch_first=True, **kwargs
    ):
        super(TimestepsPooler, self).__init__(
            feature_size, batch_first=batch_first, **kwargs
        )
        assert (
            mode is None or mode in SUPPORTED_POOLERS
        ), "Unsupported timestep pooling method.  Available methods: {SUPPORTED_POOLERS.keys()}"

        self.pooler = None

        if mode is not None:
            self.pooler = SUPPORTED_POOLERS[mode](
                feature_size, batch_first=batch_first, **kwargs
            )

    @property
    def out_size(self) -> int:
        """Define the feature size of the returned tensor

        Returns:
            int: The feature dimension of the output tensor
        """

        if self.pooler is not None:
            return cast(int, self.pooler.out_size)
        else:
            return super(TimestepsPooler, self).out_size

    def _pool(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Pool features of input tensor across timesteps

        Args:
            x (torch.Tensor): [B, L, D] Input sequence
            lengths (Optional[torch.Tensor]): Optional unpadded sequence lengths for input tensor

        Returns:
            torch.Tensor: [B, D] Output aggregated features across timesteps
        """

        if self.pooler is None:
            return x

        out: torch.Tensor = self.pooler(x, lengths=lengths)

        return out


class BaseFuser(nn.Module, metaclass=ABCMeta):
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

    def _check_n_modalities(self, n=3):
        """Check if this fuser is created for 3 modalities

        Some fusers are implemented particularly for 3 modalities (txt, audio, visual).
        Check at instantiation time if they are misused
        """

        if self.n_modalities != n:
            raise ValueError(
                f"{self.__class__.__name__} is implemented for 3 modalities, e.g. [text, audio, visual]"
            )

    @property
    @abstractmethod
    def out_size(self) -> int:
        """Output feature size.

        Each fuser specifies its output feature size
        """
        pass

    @abstractmethod
    def fuse(
        self, *mods: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Abstract method to fuse the modality representations

        Children classes should implement this method

        Args:
            *mods: List of modality tensors
            lengths (Optional[Tensor]): Lengths of each modality

        Returns:
            torch.Tensor: Fused tensor
        """
        pass

    def forward(
        self, *mods: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Fuse the modality representations

        Args:
            *mods: List of modality tensors [B, L, D]
            lengths (Optional[Tensor]): Lengths of each modality

        Returns:
            torch.Tensor: Fused tensor [B, L, self.out_size]
        """
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

    @property
    def out_size(self) -> int:
        """d_out = n_modalities * d_in

        Returns:
            int: output feature size
        """

        return self.n_modalities * self.feature_size

    def fuse(
        self, *mods: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Concatenate input tensors into a single tensor

        Example:
            fuser = CatFuser(5, 2)
            x = torch.rand(16, 6, 5)  # (B, L, D)
            y = torch.rand(16, 6, 5)  # (B, L, D)
            out = fuser(x, y)  # (B, L, 2 * D)

        Args:
            *mods: Variable number of input tensors

        Returns:
            torch.Tensor: Concatenated input tensors

        """

        return torch.cat(mods, dim=-1)


class SumFuser(BaseFuser):
    """Fuse by adding modality representations

    o = m1 + m2 + m3 ...

    Args:
        feature_size (int): Assume all modality representations have the same feature_size
        n_modalities (int): Number of input modalities
        **extra_kwargs (dict): Extra keyword arguments to maintain interoperability of children
            classes
    """

    @property
    def out_size(self) -> int:
        """d_out = d_in

        Returns:
            int: output feature size
        """

        return self.feature_size

    def fuse(
        self, *mods: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Sum input tensors into a single tensor

        Example:
            fuser = SumFuser(5, 2)
            x = torch.rand(16, 6, 5)  # (B, L, D)
            y = torch.rand(16, 6, 5)  # (B, L, D)
            out = fuser(x, y)  # (B, L, D)

        Args:
            *mods: Variable number of input tensors

        Returns:
            torch.Tensor: Summed input tensors

        """

        return torch.cat([m.unsqueeze(-1) for m in mods], dim=-1).sum(-1)


class BimodalCombinatorialFuser(BaseFuser, metaclass=ABCMeta):
    """Fuse all combinations of three modalities using a base module

    If input modalities are x, y, then the output is
    o = x || y || f(x, y)

    Where f is a network module (e.g. attention)

    Args:
        feature_size (int): Number of feature dimensions
        n_modalities (int): Number of input modalities (should be 3)
    """

    def __init__(
        self,
        feature_size: int,
        n_modalities: int,
        **kwargs,
    ):
        super(BimodalCombinatorialFuser, self).__init__(
            feature_size, n_modalities, **kwargs
        )
        self._check_n_modalities(n=2)
        self.xy = self._bimodal_fusion_module(feature_size, **kwargs)

    @abstractmethod
    def _bimodal_fusion_module(self, feature_size: int, **kwargs) -> nn.Module:
        """Module to use to fuse bimodal combinations

        Args:
            feature_size (int): Number of feature dimensions
            **kwargs: Extra kwargs to pass to nn.Module

        Returns:
            nn.Module: The module to use for fusion
        """
        pass

    @property
    def out_size(self) -> int:
        """Fused vector feature dimension

        Returns:
            int: 3 * feature_size

        """

        return 3 * self.feature_size


class BimodalBilinearFuser(BimodalCombinatorialFuser):
    def _bimodal_fusion_module(self, feature_size: int, **kwargs) -> nn.Module:
        return nn.Bilinear(feature_size, feature_size, feature_size)

    def fuse(
        self, *mods: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Perform bilinear fusion on input modalities

        Args:
            *mods: Input tensors to fuse. This module accepts 2 input modalities. [B, L, D]
            lengths (Optional[torch.Tensor]): Unpadded tensors lengths

        Returns:
            torch.Tensor: fused output vector [B, L, 3*D]

        """
        x, y = mods
        xy = self.xy(x, y)

        # B x L x 3*D
        fused = torch.cat([x, y, xy], dim=-1)

        return fused


class BimodalAttentionFuser(BimodalCombinatorialFuser):
    def _bimodal_fusion_module(self, feature_size: int, **kwargs) -> nn.Module:
        return SymmetricAttention(
            attention_size=feature_size,
            dropout=kwargs.get("dropout", 0.1),
            residual=kwargs.get("residual", True),
        )

    def fuse(
        self, *mods: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Perform attention fusion on input modalities

        Args:
            *mods: Input tensors to fuse. This module accepts 2 input modalities. [B, L, D]
            lengths (Optional[torch.Tensor]): Unpadded tensors lengths

        Returns:
            torch.Tensor: fused output vector [B, L, 3*D]

        """
        x, y = mods
        xy, yx = self.xy(x, y)
        xy = xy + yx
        # B x L x 3*D
        fused = torch.cat([x, y, xy], dim=-1)

        return fused


class TrimodalCombinatorialFuser(BaseFuser, metaclass=ABCMeta):
    """Fuse all combinations of three modalities using a base module

    If input modalities are a, t, v, then the output is
    o = t || a || v || f(t, a) || f(v, a) || f(t, v) || g(t, f(v, a)) || [ g(v, f(t,a)) ] || [g(a, f(t,v))]

    Where f and g network modules (e.g. attention) and values with [] are optional

    Args:
        feature_size (int): Number of feature dimensions
        n_modalities (int): Number of input modalities (should be 3)
        use_all_trimodal (bool): Use all optional trimodal combinations
    """

    def __init__(
        self,
        feature_size: int,
        n_modalities: int,
        use_all_trimodal: bool = False,
        **kwargs,
    ):
        super(TrimodalCombinatorialFuser, self).__init__(
            feature_size, n_modalities, **kwargs
        )
        self._check_n_modalities(n=3)
        self.use_all_trimodal = use_all_trimodal

        self.ta = self._bimodal_fusion_module(feature_size, **kwargs)
        self.va = self._bimodal_fusion_module(feature_size, **kwargs)
        self.tv = self._bimodal_fusion_module(feature_size, **kwargs)

        self.tav = self._trimodal_fusion_module(feature_size, **kwargs)

        if use_all_trimodal:
            self.vat = self._trimodal_fusion_module(feature_size, **kwargs)
            self.atv = self._trimodal_fusion_module(feature_size, **kwargs)

    @abstractmethod
    def _bimodal_fusion_module(self, feature_size: int, **kwargs) -> nn.Module:
        """Module to use to fuse bimodal combinations

        Args:
            feature_size (int): Number of feature dimensions
            **kwargs: Extra kwargs to pass to nn.Module

        Returns:
            nn.Module: The module to use for fusion
        """
        pass

    @abstractmethod
    def _trimodal_fusion_module(self, feature_size: int, **kwargs) -> nn.Module:
        """Module to use to fuse trimodal combinations

        Args:
            feature_size (int): Number of feature dimensions
            **kwargs: Extra kwargs to pass to nn.Module

        Returns:
            nn.Module: The module to use for fusion
        """
        pass

    @property
    def out_size(self) -> int:
        """Fused vector feature dimension

        Returns:
            int: 7 * feature_size if use_all_trimodal==False else 9*feature_size

        """
        multiplier = 7

        if self.use_all_trimodal:
            multiplier += 2

        return multiplier * self.feature_size


class BilinearFuser(TrimodalCombinatorialFuser):
    """Fuse all combinations of three modalities using a base module using bilinear fusion

    If input modalities are a, t, v, then the output is
    o = t || a || v || f(t, a) || f(v, a) || f(t, v) || g(t, f(v, a)) || [ g(v, f(t,a)) ] || [g(a, f(t,v))]

    Where f and g are the nn.Bilinear function and values with [] are optional

    Args:
        feature_size (int): Number of feature dimensions
        n_modalities (int): Number of input modalities (should be 3)
        use_all_trimodal (bool): Use all optional trimodal combinations
    """

    def __init__(
        self,
        feature_size: int,
        n_modalities: int,
        use_all_trimodal: bool = False,
        **kwargs,
    ):
        super(BilinearFuser, self).__init__(
            feature_size,
            n_modalities,
            use_all_trimodal=use_all_trimodal,
            **kwargs,
        )

    def _bimodal_fusion_module(self, feature_size: int, **kwargs):
        """nn.Bilinear module to use to fuse bimodal combinations

        Args:
            feature_size (int): Number of feature dimensions

        Returns:
            nn.Module: nn.Bilinear module to use for fusion
        """

        return nn.Bilinear(feature_size, feature_size, feature_size)

    def _trimodal_fusion_module(self, feature_size: int, **kwargs):
        """nn.Bilinear module to use to fuse trimodal combinations

        Args:
            feature_size (int): Number of feature dimensions

        Returns:
            nn.Module: nn.Bilinear module to use for fusion
        """

        return nn.Bilinear(feature_size, feature_size, feature_size)

    def fuse(
        self, *mods: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Perform bilinear fusion on input modalities

        Args:
            *mods: Input tensors to fuse. This module accepts 3 input modalities. [B, L, D]
            lengths (Optional[torch.Tensor]): Unpadded tensors lengths

        Returns:
            torch.Tensor: fused output vector [B, L, 7*D] or [B, L, 9*D]

        """
        txt, au, vi = mods
        ta = self.ta(txt, au)
        va = self.va(vi, au)
        tv = self.tv(txt, vi)

        tav = self.tav(txt, va)

        out_list = [txt, au, vi, ta, tv, va, tav]

        if self.use_all_trimodal:
            vat = self.vat(vi, ta)
            atv = self.atv(au, tv)

            out_list = out_list + [vat, atv]

        # B x L x 7*D or B x L x 9*D
        fused = torch.cat(out_list, dim=-1)

        return fused


class AttentionFuser(TrimodalCombinatorialFuser):
    """Fuse all combinations of three modalities using a base module using bilinear fusion

    If input modalities are a, t, v, then the output is

    Where f is SymmetricAttention and g is Attention modules and values with [] are optional
    o = t || a || v || f(t, a) || f(v, a) || f(t, v) || g(t, f(v, a)) || [ g(v, f(t,a)) ] || [g(a, f(t,v))]

    Args:
        feature_size (int): Number of feature dimensions
        n_modalities (int): Number of input modalities (should be 3)
        use_all_trimodal (bool): Use all optional trimodal combinations
        residual (bool): Use residual connection in SymmetricAttention. Defaults to True
        dropout (float): Dropout probability
    """

    def __init__(
        self,
        feature_size: int,
        n_modalities: int,
        use_all_trimodal: bool = False,
        residual: bool = True,
        dropout: float = 0.1,
        **kwargs,
    ):
        kwargs["dropout"] = dropout
        kwargs["residual"] = residual
        super(AttentionFuser, self).__init__(
            feature_size,
            n_modalities,
            use_all_trimodal=use_all_trimodal,
            **kwargs,
        )

    def _bimodal_fusion_module(self, feature_size: int, **kwargs):
        """SymmetricAttention module to fuse bimodal combinations

        Args:
            feature_size (int): Number of feature dimensions
            **kwargs: dropout and residual parameters

        Returns:
            nn.Module: SymmetricAttention module to use for fusion
        """

        return SymmetricAttention(
            attention_size=feature_size,
            dropout=kwargs.get("dropout", 0.1),
            residual=kwargs.get("residual", True),
        )

    def _trimodal_fusion_module(self, feature_size: int, **kwargs):
        """Attention module to fuse trimodal combinations

        Args:
            feature_size (int): Number of feature dimensions
            **kwargs: dropout and parameters

        Returns:
            nn.Module: Attention module to use for fusion
        """

        return Attention(
            attention_size=feature_size,
            dropout=kwargs.get("dropout", 0.1),
        )

    def fuse(
        self, *mods: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Perform attention fusion on input modalities

        Args:
            *mods: Input tensors to fuse. This module accepts 3 input modalities. [B, L, D]
            lengths (Optional[torch.Tensor]): Unpadded tensors lengths

        Returns:
            torch.Tensor: fused output vector [B, L, 7*D] or [B, L, 9*D]

        """
        txt, au, vi = mods
        ta, at = self.ta(txt, au)
        va, av = self.va(vi, au)
        tv, vt = self.tv(txt, vi)

        va = va + av
        tv = vt + tv
        ta = ta + at

        tav, _ = self.tav(txt, queries=va)

        out_list = [txt, au, vi, ta, tv, va, tav]

        if self.use_all_trimodal:
            vat, _ = self.vat(vi, queries=ta)
            atv, _ = self.atv(au, queries=tv)

            out_list = out_list + [vat, atv]

        # B x L x 7*D or B x L x 9*D
        fused = torch.cat(out_list, dim=-1)

        return fused


SUPPORTED_FUSERS = {
    "cat": CatFuser,
    "add": SumFuser,
    "sum": SumFuser,
    "bilinear": BilinearFuser,
    "attention": AttentionFuser,
}
"""Currently implemented fusers"""


def make_fuser(fusion_method: str, feature_size: int, n_modalities: int, **kwargs):
    """Helper function to instantiate a fuser given a string fusion_method parameter

    Args:
        fusion_method (str): One of the supported fusion methods [cat|add|bilinear|attention]
        feature_size (int): The input modality representations dimension
        n_modalities (int): Number of input modalities
        **kwargs: Variable keyword arguments to pass to the instantiated fuser
    """

    if fusion_method not in SUPPORTED_FUSERS.keys():
        raise NotImplementedError(
            f"The supported fusers are {SUPPORTED_FUSERS.keys()}. You provided {fusion_method}"
        )

    if fusion_method == "bilinear":
        if n_modalities == 2:
            return BimodalBilinearFuser(feature_size, n_modalities, **kwargs)
        elif n_modalities == 3:
            return BilinearFuser(feature_size, n_modalities, **kwargs)
        else:
            raise ValueError("bilinear implemented for 2 or 3 modalities")

    if fusion_method == "attention":
        if n_modalities == 2:
            return BimodalAttentionFuser(feature_size, n_modalities, **kwargs)
        elif n_modalities == 3:
            return AttentionFuser(feature_size, n_modalities, **kwargs)
        else:
            raise ValueError("attention implemented for 2 or 3 modalities")

    return SUPPORTED_FUSERS[fusion_method](feature_size, n_modalities, **kwargs)


class BaseFusionPipeline(nn.Module, metaclass=ABCMeta):
    """Base class for a fusion pipeline

    Inherit this class to implement a fusion pipeline

    """

    def __init__(self, *args, **kwargs):
        super(BaseFusionPipeline, self).__init__()

    @property
    @abstractmethod
    def out_size(self) -> int:
        """Define the feature size of the returned tensor

        Returns:
            int: The feature dimension of the output tensor
        """
        pass


class FuseAggregateTimesteps(BaseFusionPipeline):
    """Fuse input feature sequences and aggregate across timesteps

    Fuser -> TimestepsPooler

    Args:
        feature_size (int): The input modality representations dimension
        n_modalities (int): Number of input modalities
        output_size (Optional[int]): Required output size. If not provided,
            output_size = fuser.out_size
        fusion_method (str): Select which fuser to use [cat|sum|attention|bilinear]
        timesteps_pooling_method (str): TimestepsPooler method [cat|sum|rnn]
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
        timesteps_pooling_method: str = "sum",
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
        self.timesteps_pooler = TimestepsPooler(
            self.fuser.out_size,
            hidden_size=output_size,
            mode=timesteps_pooling_method,
            batch_first=batch_first,
        )

    @property
    def out_size(self) -> int:
        """Define the feature size of the returned tensor

        Returns:
            int: The feature dimension of the output tensor
        """

        return cast(int, self.timesteps_pooler.out_size)

    def forward(
        self, *mods: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Fuse the modality representations and aggregate across timesteps

        Args:
            *mods: List of modality tensors [B, L, D]
            lengths (Optional[Tensor]): Lengths of each modality

        Returns:
            torch.Tensor: Fused tensor [B, self.out_size]
        """
        fused = self.fuser(*mods, lengths=lengths)
        out: torch.Tensor = self.timesteps_pooler(fused, lengths=lengths)

        return out


class ProjectFuseAggregate(BaseFusionPipeline):
    """Project input feature sequences, fuse and aggregate across timesteps

    ModalityProjection -> Optional[ModalityWeights] -> Fuser -> TimestepsPooler

    Args:
        modality_sizes (List[int]): List of input modality representations dimensions
        projection_size (int): Project all modalities to have this feature size
        projection_type (Optional[str]): Optional projection method [linear|conv]
        fusion_method (str): Select which fuser to use [cat|sum|attention|bilinear]
        timesteps_pooling_method (str): TimestepsPooler method [cat|sum|rnn]
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
        projection_type: Optional[str] = None,
        fusion_method="cat",
        timesteps_pooling_method="sum",
        modality_weights: bool = False,
        batch_first: bool = True,
        **fuser_kwargs,
    ):
        super(ProjectFuseAggregate, self).__init__()
        n_modalities = len(modality_sizes)

        self.projection = None
        self.modality_weights = None

        if projection_type is not None:
            self.projection = ModalityProjection(
                modality_sizes, projection_size, mode=projection_type
            )

            if modality_weights:
                self.modality_weights = ModalityWeights(projection_size)

        fuser_kwargs["output_size"] = projection_size
        fuser_kwargs["fusion_method"] = fusion_method
        fuser_kwargs["timesteps_pooling_method"] = timesteps_pooling_method
        fuser_kwargs["batch_first"] = batch_first

        if "n_modalities" in fuser_kwargs:
            del fuser_kwargs["n_modalities"]

        if "projection_size" in fuser_kwargs:
            del fuser_kwargs["projection_size"]

        self.fuse_aggregate = FuseAggregateTimesteps(
            projection_size,
            n_modalities,
            **fuser_kwargs,
        )

    @property
    def out_size(self) -> int:
        """Define the feature size of the returned tensor

        Returns:
            int: The feature dimension of the output tensor
        """

        return self.fuse_aggregate.out_size

    def forward(
        self, *mods: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Project modality representations to a common dimension, fuse and aggregate across timesteps

        Optionally use modality weights

        Args:
            *mods: List of modality tensors [B, L, D]
            lengths (Optional[Tensor]): Lengths of each modality

        Returns:
            torch.Tensor: Fused tensor [B, self.out_size]
        """

        if self.projection is not None:
            mods = self.projection(*mods)

        if self.modality_weights is not None:
            mods = self.modality_weights(*mods)
        fused: torch.Tensor = self.fuse_aggregate(*mods, lengths=lengths)

        return fused
