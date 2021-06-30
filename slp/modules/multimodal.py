from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, cast

import torch
import torch.nn as nn
from slp.modules.fuse import (
    BaseFusionPipeline,
    FuseAggregateTimesteps,
    ModalityProjection,
    ProjectFuseAggregate,
)
from slp.modules.rnn import AttentiveRNN


class BaseEncoder(nn.Module, metaclass=ABCMeta):
    def __init__(self, *args, **kwargs):
        super(BaseEncoder, self).__init__()
        self.args = args
        self.kwargs = kwargs
        self.clf = None

    def _check_n_modalities(self, *mods, n=1):
        if len(mods) != n:
            raise ValueError(f"Expected {n} input modalities. You provided {len(mods)}")

    @property
    @abstractmethod
    def out_size(self) -> int:
        pass

    @abstractmethod
    def _encode(
        self, *mods: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        pass

    @abstractmethod
    def _fuse(
        self, *mods: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        pass

    def forward(
        self, *mods: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        encoded: List[torch.Tensor] = self._encode(*mods, lengths=lengths)
        fused = self._fuse(*encoded, lengths=lengths)

        return fused


class UnimodalEncoder(BaseEncoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.2,
        rnn_type: str = "lstm",
        attention: bool = True,
        aggregate_encoded: bool = False,
        **kwargs,
    ):
        super(UnimodalEncoder, self).__init__(
            input_size,
            hidden_size,
            layers=layers,
            bidirectional=bidirectional,
            dropout=dropout,
            rnn_type=rnn_type,
            attention=attention,
            **kwargs,
        )
        self.aggregate_encoded = aggregate_encoded
        self.encoder = AttentiveRNN(
            input_size,
            hidden_size,
            batch_first=True,
            layers=layers,
            merge_bi="sum",
            bidirectional=bidirectional,
            dropout=dropout,
            rnn_type=rnn_type,
            packed_sequence=True,
            attention=attention,
            return_hidden=True,
        )

    @property
    def out_size(self) -> int:
        return cast(int, self.encoder.out_size)

    def _encode(  # type: ignore
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        out, hid = self.encoder(x, lengths=lengths)

        return [out] if self.aggregate_encoded else [hid]

    def _fuse(  # type: ignore
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return x


class AudioEncoder(UnimodalEncoder):
    pass


class VisualEncoder(UnimodalEncoder):
    pass


class GloveEncoder(UnimodalEncoder):
    pass


class BimodalEncoder(BaseEncoder):
    def __init__(
        self,
        encoder1_args: Dict[str, Any],
        encoder2_args: Dict[str, Any],
        fuser_args: Dict[str, Any],
        **kwargs,
    ):
        super(BimodalEncoder, self).__init__(
            encoder1_args,
            encoder2_args,
            fuser_args,
            **kwargs,
        )
        self.input_projection = None

        if "input_projection" in fuser_args and fuser_args["input_projection"]:
            self.input_projection = ModalityProjection(
                [encoder1_args["input_size"], encoder2_args["input_size"]],
                fuser_args["hidden_size"],
                mode=fuser_args["input_projection"],
            )

        encoder1_args["return_hidden"] = True
        encoder2_args["return_hidden"] = True

        self.encoder1 = UnimodalEncoder(**encoder1_args)

        self.encoder2 = UnimodalEncoder(**encoder2_args)

        self.fuse = ProjectFuseAggregate(
            [encoder1_args["hidden_size"], encoder2_args["hidden_size"]],
            fuser_args["hidden_size"],
            **fuser_args,
        )

    @property
    def out_size(self) -> int:
        return cast(int, self.fuse.out_size)

    def _encode(
        self, *mods: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        self._check_n_modalities(*mods, n=2)
        x, y = mods

        if self.input_projection is not None:
            x, y = self.input_projection(x, y)
        x = self.encoder1(x, lengths)
        y = self.encoder2(y, lengths)

        return [x, y]

    def _fuse(
        self, *mods: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        self._check_n_modalities(*mods, n=2)
        fused: torch.Tensor = self.fuse(*mods, lengths=lengths)

        return fused


class TrimodalEncoder(BaseEncoder):
    def __init__(
        self,
        encoder1_args: Dict[str, Any],
        encoder2_args: Dict[str, Any],
        encoder3_args: Dict[str, Any],
        fuser_args: Dict[str, Any],
        **kwargs,
    ):
        super(TrimodalEncoder, self).__init__(
            encoder1_args,
            encoder2_args,
            encoder3_args,
            fuser_args,
            **kwargs,
        )
        self.input_projection = None

        if "input_projection" in fuser_args and fuser_args["input_projection"]:
            self.input_projection = ModalityProjection(
                [encoder1_args["input_size"], encoder2_args["input_size"]],
                fuser_args["hidden_size"],
                mode=fuser_args["input_projection"],
            )

        self.encoder1 = UnimodalEncoder(
            # encoder1_args["input_size"], encoder1_args["hidden_size"], **encoder1_args
            **encoder1_args
        )

        self.encoder2 = UnimodalEncoder(
            # encoder2_args["input_size"], encoder2_args["hidden_size"], **encoder2_args
            **encoder2_args
        )

        self.encoder3 = UnimodalEncoder(
            # encoder3_args["input_size"], encoder3_args["hidden_size"], **encoder3_args
            **encoder3_args
        )

        self.fuse = self._make_fusion_pipeline(
            encoder1_args,
            encoder2_args,
            encoder3_args,
            **fuser_args,
        )

    def _make_fusion_pipeline(
        self,
        encoder1_args: Dict[str, Any],
        encoder2_args: Dict[str, Any],
        encoder3_args: Dict[str, Any],
        **fuser_kwargs,
    ) -> BaseFusionPipeline:
        modality_sizes = [
            encoder1_args["hidden_size"],
            encoder2_args["hidden_size"],
            encoder3_args["hidden_size"],
        ]
        projection_size = fuser_kwargs["hidden_size"]

        return ProjectFuseAggregate(modality_sizes, projection_size, **fuser_kwargs)

    @property
    def out_size(self) -> int:
        return cast(int, self.fuse.out_size)

    def _encode(
        self, *mods: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        self._check_n_modalities(*mods, n=3)
        x, y, z = mods

        if self.input_projection is not None:
            x, y, z = self.input_projection(x, y, z)
        x = self.encoder1(x, lengths=lengths)
        y = self.encoder2(y, lengths=lengths)
        z = self.encoder3(z, lengths=lengths)

        return [x, y, z]

    def _fuse(
        self, *mods: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        self._check_n_modalities(*mods, n=3)
        fused: torch.Tensor = self.fuse(*mods, lengths=lengths)

        return fused


class MultimodalBaseline(TrimodalEncoder):
    def __init__(
        self,
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

        cfg = {
            "hidden_size": hidden_size,
            "dropout": dropout,
            "layers": encoder_layers,
            "attention": encoder_attention,
            "bidirectional": bidirectional,
            "rnn_type": rnn_type,
        }

        text_cfg = MultimodalBaseline.encoder_cfg(text_size, **cfg)
        audio_cfg = MultimodalBaseline.encoder_cfg(audio_size, **cfg)
        visual_cfg = MultimodalBaseline.encoder_cfg(visual_size, **cfg)
        fuser_cfg = MultimodalBaseline.fuser_cfg(
            hidden_size=hidden_size,
            dropout=dropout,
            residual=fuser_residual,
            use_all_trimodal=use_all_trimodal,
        )

        super(MultimodalBaseline, self).__init__(
            text_cfg, audio_cfg, visual_cfg, fuser_cfg
        )

    def _make_fusion_pipeline(
        self,
        encoder1_args: Dict[str, Any],
        encoder2_args: Dict[str, Any],
        encoder3_args: Dict[str, Any],
        **fuser_kwargs,
    ) -> BaseFusionPipeline:
        feature_size = fuser_kwargs["hidden_size"]
        n_modalities = fuser_kwargs.pop("n_modalities")

        return FuseAggregateTimesteps(feature_size, n_modalities, **fuser_kwargs)

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
        }


class MOSEIClassifier(nn.Module, metaclass=ABCMeta):
    """Encode and classify multimodal inputs

    Args:
        encoder (BaseEncoder): The encoder module
        num_classes (int): The number of target classes
        dropout (float): Dropout probability

    """

    def __init__(self, encoder: BaseEncoder, num_classes: int, dropout: float = 0.2):
        super(MOSEIClassifier, self).__init__()
        self.enc = encoder
        self.drop = nn.Dropout(p=dropout)
        self.clf = nn.Linear(self.enc.out_size, num_classes)


class UnimodalClassifier(MOSEIClassifier):
    """Encode and classify unimodal inputs

    Args:
        input_size (int): The input modality feature size
        hidden_size (int): Hidden size for RNN
        num_classes (int): The number of target classes
        layers (int): Number of RNN layers
        bidirectional (bool): Use biRNN
        dropout (float): Dropout probability
        rnn_type (str): [lstm|gru]
        attention (bool): Use attention on hidden states

    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.2,
        rnn_type: str = "lstm",
        attention: bool = True,
        **kwargs,
    ):
        enc = UnimodalEncoder(
            input_size,
            hidden_size,
            layers=layers,
            bidirectional=bidirectional,
            dropout=dropout,
            rnn_type=rnn_type,
            attention=attention,
            aggregate_encoded=True,
        )
        super(UnimodalClassifier, self).__init__(enc, num_classes)

    def forward(
        self, x: torch.Tensor, lengths: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        fused = self.enc(x, lengths=lengths["text"])
        fused = self.drop(fused)
        out: torch.Tensor = self.clf(fused)

        return out


class BimodalClassifier(MOSEIClassifier, metaclass=ABCMeta):
    def __init__(
        self,
        encoder1_args: Dict[str, Any],
        encoder2_args: Dict[str, Any],
        fuser_args: Dict[str, Any],
        num_classes: int,
        **kwargs,
    ):
        enc = BimodalEncoder(encoder1_args, encoder2_args, fuser_args, **kwargs)
        super(BimodalClassifier, self).__init__(enc, num_classes)


class AudioVisualClassifier(BimodalClassifier):
    def forward(
        self, mod_dict: Dict[str, torch.Tensor], lengths: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        mods = [mod_dict["visual"], mod_dict["audio"]]
        fused = self.enc(*mods, lengths=lengths["visual"])
        fused = self.drop(fused)
        out: torch.Tensor = self.clf(fused)

        return out


class AudioTextClassifier(BimodalClassifier):
    def forward(
        self, mod_dict: Dict[str, torch.Tensor], lengths: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        mods = [mod_dict["text"], mod_dict["audio"]]
        fused = self.enc(*mods, lengths=lengths["text"])
        fused = self.drop(fused)
        out: torch.Tensor = self.clf(fused)

        return out


class VisualTextClassifier(BimodalClassifier):
    def forward(
        self, mod_dict: Dict[str, torch.Tensor], lengths: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        mods = [mod_dict["text"], mod_dict["visual"]]
        fused = self.enc(*mods, lengths=lengths["text"])
        fused = self.drop(fused)
        out: torch.Tensor = self.clf(fused)

        return out


class TrimodalClassifier(MOSEIClassifier):
    def __init__(
        self,
        encoder1_args: Dict[str, Any],
        encoder2_args: Dict[str, Any],
        encoder3_args: Dict[str, Any],
        fuser_args: Dict[str, Any],
        num_classes: int,
        **kwargs,
    ):
        enc = TrimodalEncoder(
            encoder1_args, encoder2_args, encoder3_args, fuser_args, **kwargs
        )
        super(TrimodalClassifier, self).__init__(enc, num_classes)

    def forward(
        self, mod_dict: Dict[str, torch.Tensor], lengths: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        mods = [mod_dict["text"], mod_dict["audio"], mod_dict["visual"]]
        fused = self.enc(*mods, lengths=lengths["text"])
        fused = self.drop(fused)
        out: torch.Tensor = self.clf(fused)

        return out


class MultimodalBaselineClassifier(MOSEIClassifier):
    def __init__(
        self,
        num_classes: int = 1,
        text_size: int = 300,
        audio_size: int = 74,
        visual_size: int = 35,
        hidden_size: int = 100,
    ):
        enc = MultimodalBaseline(
            text_size=text_size,
            audio_size=audio_size,
            visual_size=visual_size,
            hidden_size=hidden_size,
        )
        super(MultimodalBaselineClassifier, self).__init__(enc, num_classes)

    def forward(
        self, mod_dict: Dict[str, torch.Tensor], lengths: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        mods = [mod_dict["text"], mod_dict["audio"], mod_dict["visual"]]
        fused = self.enc(*mods, lengths=lengths["text"])
        fused = self.drop(fused)
        out: torch.Tensor = self.clf(fused)

        return out
