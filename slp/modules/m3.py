from typing import Any, Dict, List, Optional

import torch
from slp.modules.fuse import BaseFusionPipeline, FuseAggregateTimesteps
from slp.modules.mmdrop import MultimodalDropout
from slp.modules.multimodal import MOSEIClassifier, MultimodalBaseline


class M3FuseAggregate(BaseFusionPipeline):
    """MultimodalDropout, Fuse input feature sequences and aggregate across timesteps

    MultimodalDropout -> Fuser -> TimestepsPooler

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
        mmdrop_prob: float = 0.2,
        mmdrop_individual_mod_prob: Optional[List[float]] = None,
        mmdrop_algorithm: str = "hard",
        **fuser_kwargs,
    ):
        super(M3FuseAggregate, self).__init__()

        self.m3 = MultimodalDropout(
            p=mmdrop_prob,
            n_modalities=n_modalities,
            p_mod=mmdrop_individual_mod_prob,
            mode=mmdrop_algorithm,
        )

        fuser_kwargs["output_size"] = output_size
        fuser_kwargs["fusion_method"] = fusion_method
        fuser_kwargs["timesteps_pooling_method"] = timesteps_pooling_method
        fuser_kwargs["batch_first"] = batch_first

        if "n_modalities" in fuser_kwargs:
            fuser_kwargs.pop("n_modalities")  # Avoid multiple arguments

        if "projection_size" in fuser_kwargs:
            fuser_kwargs.pop("projection_size")  # Avoid multiple arguments

        self.fuse_aggregate = FuseAggregateTimesteps(
            feature_size,
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
        """Fuse the modality representations and aggregate across timesteps

        Args:
            *mods: List of modality tensors [B, L, D]
            lengths (Optional[Tensor]): Lengths of each modality

        Returns:
            torch.Tensor: Fused tensor [B, self.out_size]
        """
        mods_masked: List[torch.Tensor] = self.m3(*mods)
        fused: torch.Tensor = self.fuse_aggregate(*mods_masked, lengths=lengths)

        return fused


class M3(MultimodalBaseline):
    def _make_fusion_pipeline(
        self,
        encoder1_args: Dict[str, Any],
        encoder2_args: Dict[str, Any],
        encoder3_args: Dict[str, Any],
        **fuser_kwargs,
    ) -> BaseFusionPipeline:

        feature_size = fuser_kwargs.pop("hidden_size")
        n_modalities = fuser_kwargs.pop("n_modalities")

        return M3FuseAggregate(feature_size, n_modalities, **fuser_kwargs)

    @staticmethod
    def encoder_cfg(input_size: int, **cfg) -> Dict[str, Any]:
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
        return {
            "n_modalities": 3,
            "dropout": cfg.get("dropout", 0.2),
            "output_size": cfg.get("hidden_size", 100),
            "hidden_size": cfg.get("hidden_size", 100),
            "fusion_method": "attention",
            "timesteps_pooling_method": "rnn",
            "residual": cfg.get("residual", True),
            "use_all_trimodal": cfg.get("use_all_trimodal", True),
            "mmdrop_prob": 0.2,
            "mmdrop_individual_mod_prob": None,
            "mmdrop_algorithm": "hard",
        }


class M3Classifier(MOSEIClassifier):
    def __init__(
        self,
        num_classes: int = 1,
        text_size: int = 300,
        audio_size: int = 74,
        visual_size: int = 35,
        hidden_size: int = 100,
        dropout: float = 0.2,
        encoder_layers: float = 1,
        bidirectional: bool = True,
        rnn_type: str = "lstm",
        encoder_attention: bool = True,
        fuser_residual: bool = True,
        use_all_trimodal: bool = False,
    ):
        enc = M3(
            text_size=text_size,
            audio_size=audio_size,
            visual_size=visual_size,
            hidden_size=hidden_size,
            dropout=dropout,
            encoder_layers=encoder_layers,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
            encoder_attention=encoder_attention,
            fuser_residual=fuser_residual,
            use_all_trimodal=use_all_trimodal,
        )
        super(M3Classifier, self).__init__(enc, num_classes, dropout=dropout)

    def forward(
        self, mod_dict: Dict[str, torch.Tensor], lengths: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        mods = [mod_dict["text"], mod_dict["audio"], mod_dict["visual"]]
        fused = self.enc(*mods, lengths=lengths["text"])
        fused = self.drop(fused)
        out: torch.Tensor = self.clf(fused)

        return out
