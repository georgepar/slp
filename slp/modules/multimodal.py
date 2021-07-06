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
        """Base class implementing a multimodal encoder

        A BaseEncoder child encodes and fuses the modality  features
        and returns representations ready to be provided to a classification layer
        """
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
        """An encoder returns its output size

        Returns:
            int: The output feature size of the encoder
        """
        pass

    @abstractmethod
    def _make_fusion_pipeline(
        self,
        encoder_output_sizes: List[int],
        **fuser_kwargs,
    ) -> BaseFusionPipeline:
        """Create a fusion pipeline. This module is used for fusion.

        Abstract method to be implemented by subclasses

        Args:
            encoder_output_sizes (List[int]): Output feature sizes of the internal encoders
            **fuser_kwargs: Variable keyword argumets to be passed to the fusion pipeline


        Returns:
            BaseFusionPipeline: The fusion pipeline to use
        """
        pass

    @abstractmethod
    def _encode(
        self, *mods: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """Encode method

        Encode the input modalities into a latent space

        Args:
            *mods (torch.Tensor): Variable input modality tensors [B, L, D]
            lengths (Optional[torch.Tensor], optional): The unpadded tensor lengths. Defaults to None.

        Returns:
            List[torch.Tensor]: The encoded modality tensors [B, L, D]
        """
        pass

    @abstractmethod
    def _fuse(
        self, *mods: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Fuse method

        Fuse input modalities into a single vector

        Args:
            *mods (torch.Tensor): Variable input modality tensors [B, L, D]
            lengths (Optional[torch.Tensor], optional): The unpadded tensor lengths. Defaults to None.

        Returns:
            torch.Tensor: The fused tensor [B, D]
        """
        pass

    def forward(
        self, *mods: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode + fuse

        Args:
            *mods (torch.Tensor): Variable input modality tensors [B, L, D]
            lengths (Optional[torch.Tensor], optional): The unpadded tensor lengths. Defaults to None.

        Returns:
            torch.Tensor: The fused tensor [B, D]
        """
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
        merge_bi: str = "sum",
        aggregate_encoded: bool = False,
        **kwargs,
    ):
        """Single modality encoder

        Encode a single modality using an Attentive RNN

        Args:
            input_size (int): Input feature size
            hidden_size (int): RNN hidden size
            layers (int, optional): Number of RNN layers. Defaults to 1.
            bidirectional (bool, optional): Use bidirectional RNN. Defaults to True.
            dropout (float, optional): Dropout probability. Defaults to 0.2.
            rnn_type (str, optional): lstm or gru. Defaults to "lstm".
            attention (bool, optional): Use attention over hidden states. Defaults to True.
            merge_bi (str, optional): How to merge hidden states [sum|cat]. Defaults to sum.
            aggregate_encoded (bool, optional): Aggregate hidden states. Defaults to False.
        """
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
            merge_bi=merge_bi,
            bidirectional=bidirectional,
            dropout=dropout,
            rnn_type=rnn_type,
            packed_sequence=True,
            attention=attention,
            return_hidden=True,
        )

    def _make_fusion_pipeline(
        self,
        encoder_output_sizes: List[int],
        **fuser_kwargs,
    ) -> BaseFusionPipeline:
        """Create a fusion pipeline. This module is used for fusion.

        Abstract method to be implemented by subclasses

        Args:
            encoder_output_sizes (List[int]): Output feature sizes of the internal encoders
            **fuser_kwargs: Variable keyword argumets to be passed to the fusion pipeline


        Returns:
            BaseFusionPipeline: The fusion pipeline to use
        """
        pass

    @property
    def out_size(self) -> int:
        """Output feature size

        Returns:
            int: Output feature size
        """
        return cast(int, self.encoder.out_size)

    def _encode(  # type: ignore
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """Encode input features using an attentive RNN

        Args:
            x (torch.Tensor): Input features [B, L, D]
            lengths (Optional[torch.Tensor], optional): Unpadded sequence lengths. Defaults to None.

        Returns:
            List[torch.Tensor]:
                * aggregate_encoded == True: Weighted sum of hidden states using attention weights [B, D]
                * aggregate_encoded == False: Weighted hidden states [B, L, D]
        """
        out, hid = self.encoder(x, lengths=lengths)

        return [out] if self.aggregate_encoded else [hid]

    def _fuse(  # type: ignore
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Fuse method. Here we have one modality, therefore it is identity

        Args:
            *mods (torch.Tensor): Variable input modality tensors [B, L, D] or [B, D]
            lengths (Optional[torch.Tensor], optional): The unpadded tensor lengths. Defaults to None.

        Returns:
            torch.Tensor: The fused tensor [B, L, D] or [B, D]
        """
        return x


class AudioEncoder(UnimodalEncoder):
    """Alias for Unimodal Encoder"""

    pass


class VisualEncoder(UnimodalEncoder):
    """Alias for Unimodal Encoder"""

    pass


class GloveEncoder(UnimodalEncoder):
    """Alias for Unimodal Encoder"""

    pass


class BimodalEncoder(BaseEncoder):
    def __init__(
        self,
        encoder1_args: Dict[str, Any],
        encoder2_args: Dict[str, Any],
        fuser_args: Dict[str, Any],
        **kwargs,
    ):
        """Two modality encoder

        Encode + Fuse two input modalities

        Example encoder_args:
            {
                "input_size": 35,
                "hidden_size": 100,
                "layers": 1,
                "bidirectional": True,
                "dropout": 0.2,
                "rnn_type": "lstm",
                "attention": True,
            }

        Example fuser_args:
            {
                "n_modalities": 3,
                "dropout": 0.2,
                "output_size": 100,
                "hidden_size": 100,
                "fusion_method": "cat",
                "timesteps_pooling_method": "rnn",
            }

        Args:
            encoder1_args (Dict[str, Any]): Configuration for first encoder
            encoder2_args (Dict[str, Any]): Configuration for second encoder
            fuser_args (Dict[str, Any]): Configuration for fuser
        """
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

        self.fuse = self._make_fusion_pipeline(
            [self.encoder1.out_size, self.encoder2.out_size], **fuser_args
        )

    def _make_fusion_pipeline(
        self,
        encoder_output_sizes: List[int],
        **fuser_kwargs,
    ) -> BaseFusionPipeline:
        """Create a Project fuse and aggregate pipeline for fusion

        Args:
            encoder_output_sizes (List[int]): Output sizes of the unimodal encoders

        Returns:
            BaseFusionPipeline: ProjectFuseAggregate instance, configured using **fuser_kwargs
        """
        projection_size = fuser_kwargs["hidden_size"]

        return ProjectFuseAggregate(
            encoder_output_sizes, projection_size, **fuser_kwargs
        )

    @property
    def out_size(self) -> int:
        """Output feature size

        Returns:
            int: Output feature size
        """
        return cast(int, self.fuse.out_size)

    def _encode(
        self, *mods: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """Encode method

        Encode two input modalities into a latent space

        Args:
            *mods (torch.Tensor): Variable input modality tensors [B, L, D]
            lengths (Optional[torch.Tensor], optional): The unpadded tensor lengths. Defaults to None.

        Returns:
            List[torch.Tensor]: The encoded modality tensors [B, L, D]
        """
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
        """Fuse method. Here we have one modality, therefore it is identity

        Args:
            *mods (torch.Tensor): Variable input modality tensors [B, L, D] or [B, D]
            lengths (Optional[torch.Tensor], optional): The unpadded tensor lengths. Defaults to None.

        Returns:
            torch.Tensor: The fused tensor [B, L, D] or [B, D]
        """
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
        """Two modality encoder

        Encode + Fuse three input modalities

        Example encoder_args:
            {
                "input_size": 35,
                "hidden_size": 100,
                "layers": 1,
                "bidirectional": True,
                "dropout": 0.2,
                "rnn_type": "lstm",
                "attention": True,
            }

        Example fuser_args:
            {
                "n_modalities": 3,
                "dropout": 0.2,
                "output_size": 100,
                "hidden_size": 100,
                "fusion_method": "cat",
                "timesteps_pooling_method": "rnn",
            }

        Args:
            encoder1_args (Dict[str, Any]): Configuration for first encoder
            encoder2_args (Dict[str, Any]): Configuration for second encoder
            encoder3_args (Dict[str, Any]): Configuration for third encoder
            fuser_args (Dict[str, Any]): Configuration for fuser
        """
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

        self.encoder1 = UnimodalEncoder(**encoder1_args)

        self.encoder2 = UnimodalEncoder(**encoder2_args)

        self.encoder3 = UnimodalEncoder(**encoder3_args)
        # encoder3_args["input_size"], encoder3_args["hidden_size"], **encoder3_args

        self.fuse = self._make_fusion_pipeline(
            [self.encoder1.out_size, self.encoder2.out_size, self.encoder3.out_size],
            **fuser_args,
        )

    def _make_fusion_pipeline(
        self,
        encoder_output_sizes: List[int],
        **fuser_kwargs,
    ) -> BaseFusionPipeline:
        """Create a Project fuse and aggregate pipeline for fusion

        Args:
            encoder_output_sizes (List[int]): Output sizes of the unimodal encoders

        Returns:
            BaseFusionPipeline: ProjectFuseAggregate instance, configured using **fuser_kwargs
        """
        projection_size = fuser_kwargs["hidden_size"]

        return ProjectFuseAggregate(
            encoder_output_sizes, projection_size, **fuser_kwargs
        )

    @property
    def out_size(self) -> int:
        """Output feature size

        Returns:
            int: Output feature size
        """
        return cast(int, self.fuse.out_size)

    def _encode(
        self, *mods: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """Encode method

        Encode three input modalities into a latent space

        Args:
            *mods (torch.Tensor): Variable input modality tensors [B, L, D]
            lengths (Optional[torch.Tensor], optional): The unpadded tensor lengths. Defaults to None.

        Returns:
            List[torch.Tensor]: The encoded modality tensors [B, L, D]
        """
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
        """Fuse method. Here we have one modality, therefore it is identity

        Args:
            *mods (torch.Tensor): Variable input modality tensors [B, L, D] or [B, D]
            lengths (Optional[torch.Tensor], optional): The unpadded tensor lengths. Defaults to None.

        Returns:
            torch.Tensor: The fused tensor [B, L, D] or [B, D]
        """
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
        merge_bi: str = "sum",
        rnn_type: str = "lstm",
        encoder_attention: bool = True,
        fuser_residual: bool = True,
        use_all_trimodal: bool = False,
    ):
        """Multimodal baseline architecture

        This baseline composes of three unimodal RNNs followed by an Attention Fuser and an RNN timestep pooler.
        The default configuration is tuned for good performance on MOSEI.

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
        """
        cfg = {
            "hidden_size": hidden_size,
            "dropout": dropout,
            "layers": encoder_layers,
            "attention": encoder_attention,
            "bidirectional": bidirectional,
            "rnn_type": rnn_type,
            "merge_bi": merge_bi,
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
        encoder_output_sizes: List[int],
        **fuser_kwargs,
    ) -> BaseFusionPipeline:
        """Fuse and Aggregate timesteps

        Args:
            encoder_output_sizes (List[int]): Output feature sizes of the unimodal encoders

        Returns:
            BaseFusionPipeline: FuseAggregateTimesteps instance, configured using **fuser_kwargs
        """
        feature_size = encoder_output_sizes[0]
        n_modalities = fuser_kwargs.pop("n_modalities")

        assert all(
            s == encoder_output_sizes[0] for s in encoder_output_sizes
        ), f"{self.__class__.__name__} All encoders should have the same output size"
        assert (
            n_modalities == 3
        ), f"{self.__class__.__name__} implemented for 3 input modalities"

        return FuseAggregateTimesteps(feature_size, n_modalities, **fuser_kwargs)

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
            "merge_bi": cfg.get("merge_bi", "sum"),
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


class MOSEIClassifier(nn.Module, metaclass=ABCMeta):
    def __init__(self, encoder: BaseEncoder, num_classes: int, dropout: float = 0.2):
        """Encode and classify multimodal inputs

        Args:
            encoder (BaseEncoder): The encoder module
            num_classes (int): The number of target classes
            dropout (float): Dropout probability

        """
        super(MOSEIClassifier, self).__init__()
        self.enc = encoder
        self.drop = nn.Dropout(p=dropout)
        self.clf = nn.Linear(self.enc.out_size, num_classes)


class UnimodalClassifier(MOSEIClassifier):
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
        dropout: float = 0.2,
        encoder_layers: float = 1,
        bidirectional: bool = True,
        rnn_type: str = "lstm",
        encoder_attention: bool = True,
        fuser_residual: bool = True,
        use_all_trimodal: bool = False,
    ):
        enc = MultimodalBaseline(
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
        super(MultimodalBaselineClassifier, self).__init__(
            enc, num_classes, dropout=dropout
        )

    def forward(
        self, mod_dict: Dict[str, torch.Tensor], lengths: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        mods = [mod_dict["text"], mod_dict["audio"], mod_dict["visual"]]
        fused = self.enc(*mods, lengths=lengths["text"])
        fused = self.drop(fused)
        out: torch.Tensor = self.clf(fused)

        return out
