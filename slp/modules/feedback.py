import abc
from typing import List, Mapping, Optional, Type

import torch
import torch.nn as nn
from slp.modules.feedforward import TwoLayer
from slp.modules.rnn import AttentiveRNN
from torch.nn.modules.container import T


class BaseFeedbackUnit(nn.Module, metaclass=abc.ABCMeta):
    def __init__(
        self, top_size: int, target_size: int, n_top_modalities: int, **kwargs
    ):
        """Base class for feedback unit

        Feedback units are responsible for projecting top-level crossmodal
        representations to bottom-level features and applying the top-down masks

        Args:
            top_size (int): Feature size of the top-level representations
            target_size (int): Feature size of the bottom-level features
            n_top_modalities (int): Number of modalities to use for feedback
        """
        super(BaseFeedbackUnit, self).__init__()
        self.n_ = n_top_modalities

        self.mask_layers = nn.ModuleList(
            [
                self.make_mask_layer(top_size, target_size, **kwargs)
                for _ in range(self.n_)
            ]
        )

    @abc.abstractmethod
    def make_mask_layer(self, top_size: int, target_size: int, **kwargs) -> nn.Module:
        """Abstract method to instantiate the layer to use for top-down feedback

        To be implemented by subclasses

        Args:
            top_size (int): Feature size of the top-level representations
            target_size (int): Feature size of the bottom-level features
            **kwargs: extra configuration for the feedback layer

        Returns:
            nn.Module: The instanstiated feedback layer
        """
        pass

    def _get_feedback_mask_one_modality(
        self,
        m_top: torch.Tensor,
        modality_index: int,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Create the feedback mask for one top modality

        $$m = \sigma(L(m^i_{top}))$$

        where $L$ is the feedback layer, $m^i_{top}$ are the top level representations for
        modality $i$ and $\sigma$ is the sigmoid activation.

        Args:
            m_top (torch.Tensor): Torch tensor containing the top level features [B, L, top_size]
            modality_index (int): Index to choose which layer in the ModuleList to use.
            lengths (Optional[torch.Tensor], optional): Original unpadded lengths [B]. Defaults to None.

        Returns:
            torch.Tensor: Top-down mask given one top modality [B, L, target_size]
        """
        return torch.sigmoid(self.mask_layers[modality_index](m_top))

    def _get_feedback_mask(
        self, *mods_top: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""Get the feedback mask given all the top-level representations

        $$m = \frac{1}{n}\sum_{i=1}^{n}[\sigma(L(m^i_{top}))]$$

        where $L$ is the feedback layer, $m^i_{top}$ are the top level representations for
        modality $i$ and $\sigma$ is the sigmoid activation.

        Args:
            *mods_top (torch.Tensor): The top-level representations for each modality [B, L, top_size]
            lengths (Optional[torch.Tensor], optional): Original unpadded tensor lengths. Defaults to None.

        Raises:
            ValueError: When number of modalities passed through mods_top is different than the number of layers in self.mask_layers

        Returns:
            (torch.Tensor): Top-down feedback mask [B, L, target_size]
        """
        if len(mods_top) != len(self.mask_layers):
            raise ValueError(f"Invalid number of modalities passed. Expected {self.n_}")

        # [(B, L, D), (B, L, D), (B, L, D)] -> (B, L, D, 3).sum(-1) -> (B,L,D)
        mask = (1.0 / self.n_) * torch.cat(
            [
                self._get_feedback_mask_one_modality(m, i, lengths=lengths).unsqueeze(
                    -1
                )
                for i, m in enumerate(mods_top)
            ],
            dim=-1,
        ).sum(-1)

        return mask

    def forward(
        self,
        x_bottom: torch.Tensor,
        *mods_top: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the top-down masks to the input feature vector

        x = x * top_down_mask

        Args:
            x_bottom (torch.Tensor): Bottom-level features [B, L, target_size]
            *mods_top (torch.Tensor): Top-level modality representations
            lengths (Optional[torch.Tensor], optional): Original unpadded tensor lengths. Defaults to None.

        Returns:
            torch.Tensor: Masked low level feature tensor [B, L, target_size]
        """
        mask = self._get_feedback_mask(*mods_top, lengths=lengths)
        x_bottom = x_bottom * mask

        return x_bottom


class GatedFeedbackUnit(BaseFeedbackUnit):
    r"""Apply feedback mask using simple gating mechanism

    $$x_bottom = x_bottom * \frac{1}{2} [\sigma(W1 * y_top) + \sigma(W2 * z_top)]$$

    """

    def make_mask_layer(self, top_size: int, target_size: int, **kwargs) -> nn.Module:
        """Use a simple nn.Linear layer for top-down projection


        Args:
            top_size (int): Feature size of the top-level representations
            target_size (int): Feature size of the bottom-level features
            **kwargs: extra configuration for the feedback layer

        Returns:
            nn.Module: nn.Linear instance with dropout
        """
        return nn.Sequential(
            nn.Linear(top_size, target_size),
            nn.Dropout(p=kwargs.get("dropout", 0.2)),
        )


class RNNFeedbackUnit(BaseFeedbackUnit):
    r"""Apply feedback mask using top-down RNN layers

    $$x_bottom = x_bottom * \frac{1}{2} [\sigma(RNN(y_top)) + \sigma(RNN(z_top))]$$

    """

    def make_mask_layer(self, top_size: int, target_size: int, **kwargs) -> nn.Module:
        """Use an RNN for top-down projection


        Args:
            top_size (int): Feature size of the top-level representations
            target_size (int): Feature size of the bottom-level features
            **kwargs: extra configuration for the feedback layer

        Returns:
            nn.Module: slp.modules.rnn.AttentiveRNN instance
        """
        return AttentiveRNN(
            top_size,
            hidden_size=target_size,
            attention=kwargs.get("attention", False),
            dropout=kwargs.get("dropout", 0.2),
            return_hidden=True,
            bidirectional=kwargs.get("bidirectional", False),
            merge_bi="sum",
            rnn_type=kwargs.get("rnn_type", "lstm"),
        )

    def _get_feedback_mask_one_modality(self, m_top, modality_index, lengths=None):
        """Modified implementation to pass the lengths to RNN forward

        $$m = \sigma(L(m^i_{top}))$$

        where $L$ is an RNN, $m^i_{top}$ are the top level representations for
        modality $i$ and $\sigma$ is the sigmoid activation.

        Args:
            m_top (torch.Tensor): Torch tensor containing the top level features [B, L, top_size]
            modality_index (int): Index to choose which layer in the ModuleList to use.
            lengths (Optional[torch.Tensor], optional): Original unpadded lengths [B]. Defaults to None.

        Returns:
            torch.Tensor: Top-down mask given one top modality [B, L, target_size]
        """
        _, m = self.mask_layers[modality_index](m_top, lengths)

        return torch.sigmoid(m)


class BoomFeedbackUnit(BaseFeedbackUnit):
    def make_mask_layer(self, top_size, target_size, **kwargs):
        """Use an boom module for top-down projection

        A boom module is a two-layer MLP where the inner projection size is
        much larger than the input and output size. (similar to Position feedforward in transformers)

        Args:
            top_size (int): Feature size of the top-level representations
            target_size (int): Feature size of the bottom-level features
            **kwargs: extra configuration for the feedback layer

        Returns:
            nn.Module: slp.modules.feedforward.TwoLayer instance
        """
        return TwoLayer(
            top_size,
            2 * top_size,
            target_size,
            activation=kwargs.get("activation", "gelu"),
            dropout=kwargs.get("dropout", 0.2),
        )


class DownUpFeedbackUnit(BaseFeedbackUnit):
    def make_mask_layer(self, top_size, target_size, **kwargs):
        """Use an down-up module for top-down projection

        A down-up module is a two-layer MLP where the inner projection size is
        much smaller than the input and output size. (Similar to adapyers)

        Args:
            top_size (int): Feature size of the top-level representations
            target_size (int): Feature size of the bottom-level features
            **kwargs: extra configuration for the feedback layer

        Returns:
            nn.Module: slp.modules.feedforward.TwoLayer instance
        """
        return TwoLayer(
            top_size,
            top_size // 5,
            target_size,
            activation=kwargs.get("activation", "gelu"),
            dropout=kwargs.get("dropout", 0.2),
        )


SUPPORTED_FEEDBACK_UNITS: Mapping[str, Type[BaseFeedbackUnit]] = {
    "gated": GatedFeedbackUnit,
    "sigmoid": GatedFeedbackUnit,
    "rnn": RNNFeedbackUnit,
    "downup": DownUpFeedbackUnit,
    "boom": BoomFeedbackUnit,
}


def _make_feedback_unit(
    top_size: int,
    target_size: int,
    n_top_modalities: int,
    mask_type: str = "rnn",
    **kwargs,
) -> BaseFeedbackUnit:
    if mask_type not in SUPPORTED_FEEDBACK_UNITS:
        raise ValueError(f"Supported mask types: {SUPPORTED_FEEDBACK_UNITS.keys()}")

    return SUPPORTED_FEEDBACK_UNITS[mask_type](
        top_size, target_size, n_top_modalities, **kwargs
    )


class Feedback(nn.Module):
    def __init__(
        self,
        top_size: int,
        bottom_modality_sizes: List[int],
        use_self: bool = False,
        mask_type: str = "rnn",
        **kwargs,
    ):
        """Feedback module

        Given a list of low-level features and top-level representations for n modalities:

        * Create top-down masks for each modality
        * Apply top-down masks to the low level features
        * Return masked low-level features

        Args:
            top_size (int): Feature size for top-level representations (Common across modalities)
            bottom_modality_sizes (List[int]): List of feature sizes for each low-level modality feature
            use_self (bool, optional): Include the self modality when creating the top-down mask. Defaults to False.
            mask_type (str, optional): Which feedback unit to use [rnn|gated|boom|downup]. Defaults to "rnn".
        """
        super(Feedback, self).__init__()

        n_top_modalities = len(bottom_modality_sizes)
        self.use_self = use_self

        if not use_self:
            n_top_modalities = n_top_modalities - 1

        self.feedback_units = nn.ModuleList(
            [
                _make_feedback_unit(
                    top_size,
                    bottom_modality_sizes[i],
                    n_top_modalities,
                    mask_type=mask_type,
                    **kwargs,
                )
                for i in range(len(bottom_modality_sizes))
            ]
        )

    def forward(
        self,
        mods_bottom: List[torch.Tensor],
        mods_top: List[torch.Tensor],
        lengths: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Create and apply the top-down masks to mods_bottom

        Args:
            mods_bottom (List[torch.Tensor]): Low-level features for each modality
            mods_top (List[torch.Tensor]): High-level representations for each modality
            lengths (Optional[torch.Tensor], optional): Original unpadded sequence lengths. Defaults to None.

        Returns:
            List[torch.Tensor]: Masked low level features for each modality
        """
        out = []

        for i, bm in enumerate(mods_bottom):
            top = mods_top if self.use_self else mods_top[:i] + mods_top[i + 1 :]
            masked = self.feedback_units[i](bm, *top, lengths=lengths)
            out.append(masked)

        return out
