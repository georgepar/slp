from typing import Any, Dict, List, Optional

import torch
from slp.modules.feedback import Feedback
from slp.modules.fuse import FuseAggregateTimesteps
from slp.modules.mmdrop import MultimodalDropout
from slp.modules.multimodal import MOSEIClassifier, MultimodalBaseline


class MMLatch(MultimodalBaseline):
    def __init__(
        self,
        text_size: int = 300,
        audio_size: int = 74,
        visual_size: int = 35,
        hidden_size: int = 100,
        dropout: float = 0.2,
        encoder_layers: float = 1,
        bidirectional: bool = True,
        merge_bi: str = "sum",
        rnn_type: str = "lstm",
        encoder_attention: bool = True,
        fuser_residual: bool = True,
        use_all_trimodal: bool = False,
        feedback: bool = True,
        use_self_feedback: bool = False,
        feedback_algorithm: str = "rnn",
    ):
        """MMLatch implementation

        Multimodal baseline + feedback

        Args:
            text_size (int, optional): Text input size. Defaults to 300.
            audio_size (int, optional): Audio input size. Defaults to 74.
            visual_size (int, optional): Visual input size. Defaults to 35.
            hidden_size (int, optional): Hidden dimension. Defaults to 100.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
            encoder_layers (float, optional): Number of encoder layers. Defaults to 1.
            bidirectional (bool, optional): Use bidirectional RNNs. Defaults to True.
            merge_bi (str, optional): Bidirectional merging method in the encoders. Defaults to "sum".
            rnn_type (str, optional): RNN type [lstm|gru]. Defaults to "lstm".
            encoder_attention (bool, optional): Use attention in the encoder RNNs. Defaults to True.
            fuser_residual (bool, optional): Use vilbert like residual in the attention fuser. Defaults to True.
            use_all_trimodal (bool, optional): Use all trimodal interactions for the Attention fuser. Defaults to False.
            feedback (bool, optional): Use top-down feedback. Defaults to True.
            use_self_feedback (bool, optional): If false use only crossmodal features for top-down feedback. If True also use the self modality. Defaults to False.
            feedback_algorithm (str, optional): Feedback module [rnn|boom|gated|downup]. Defaults to "rnn".
        """
        super(MMLatch, self).__init__(
            text_size=text_size,
            audio_size=audio_size,
            visual_size=visual_size,
            hidden_size=hidden_size,
            dropout=dropout,
            encoder_layers=encoder_layers,
            bidirectional=bidirectional,
            merge_bi=merge_bi,
            rnn_type=rnn_type,
            encoder_attention=encoder_attention,
            fuser_residual=fuser_residual,
            use_all_trimodal=use_all_trimodal,
        )

        self.feedback = None

        if feedback:
            self.feedback = Feedback(
                hidden_size,
                [text_size, audio_size, visual_size],
                use_self=use_self_feedback,
                mask_type=feedback_algorithm,
            )

    @staticmethod
    def encoder_cfg(input_size: int, **cfg) -> Dict[str, Any]:
        """Static method to create the encoder configuration

        The default configuration is provided here
        This configuration corresponds to the official paper implementation
        and is tuned for CMU MOSEI.

        Args:
            input_size (int): Input modality size
            **cfg: Optional keyword arguments

        Returns:
            Dict[str, Any]: The encoder configuration
        """
        return {
            "input_size": input_size,
            "hidden_size": cfg.get("hidden_size", 100),
            "layers": cfg.get("layers", 1),
            "bidirectional": cfg.get("bidirectional", True),
            "dropout": cfg.get("dropout", 0.2),
            "rnn_type": cfg.get("rnn_type", "lstm"),
            "attention": cfg.get("attention", True),
        }

    @staticmethod
    def fuser_cfg(**cfg) -> Dict[str, Any]:
        """Static method to create the fuser configuration

        The default configuration is provided here
        This configuration corresponds to the official paper implementation
        and is tuned for CMU MOSEI.

        Args:
            **cfg: Optional keyword arguments

        Returns:
            Dict[str, Any]: The fuser configuration
        """
        return {
            "n_modalities": 3,
            "dropout": cfg.get("dropout", 0.2),
            "output_size": cfg.get("hidden_size", 100),
            "hidden_size": cfg.get("hidden_size", 100),
            "fusion_method": "attention",
            "timesteps_pooling_method": "rnn",
            "residual": cfg.get("residual", True),
            "use_all_trimodal": cfg.get("use_all_trimodal", True),
        }

    def forward(
        self, *mods: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        encoded: List[torch.Tensor] = self._encode(*mods, lengths=lengths)

        if self.feedback is not None:
            mods_feedback: List[torch.Tensor] = self.feedback(
                mods, encoded, lengths=lengths
            )
            encoded = self._encode(*mods_feedback, lengths=lengths)

        fused = self._fuse(*encoded, lengths=lengths)

        return fused


class MMLatchClassifier(MOSEIClassifier):
    def __init__(
        self,
        num_classes: int = 1,
        text_size: int = 300,
        audio_size: int = 74,
        visual_size: int = 35,
        hidden_size: int = 100,
    ):
        enc = MMLatch(
            text_size=text_size,
            audio_size=audio_size,
            visual_size=visual_size,
            hidden_size=hidden_size,
        )
        super(MMLatchClassifier, self).__init__(enc, num_classes)

    def forward(
        self, mod_dict: Dict[str, torch.Tensor], lengths: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        mods = [mod_dict["text"], mod_dict["audio"], mod_dict["visual"]]
        fused = self.enc(*mods, lengths=lengths["text"])
        fused = self.drop(fused)
        out: torch.Tensor = self.clf(fused)

        return out
