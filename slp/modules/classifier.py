from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from slp.modules.embed import PositionalEncoding
from slp.modules.rnn import AttentiveRNN, TokenRNN
from slp.modules.transformer import (
    TransformerSequenceEncoder,
    TransformerTokenSequenceEncoder,
)


class Classifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        encoded_features: int,
        num_classes: int,
        dropout: float = 0.2,
    ):
        """Classifier wrapper module

        Stores a Neural Network encoder and adds a classification layer on top.

        Args:
            encoder (nn.Module): [description]
            encoded_features (int): [description]
            num_classes (int): [description]
            dropout (float): Drop probability
        """
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.drop = nn.Dropout(dropout)
        self.clf = nn.Linear(encoded_features, num_classes)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Encode inputs using the encoder network and perform classification

        Returns:
            torch.Tensor: [B, *, num_classes] Logits tensor
        """
        encoded: torch.Tensor = self.encoder(*args, **kwargs)  # type: ignore
        out: torch.Tensor = self.drop(encoded)
        out = self.clf(out)

        return out


# class SequenceClassifier(Classifier):
#     def forward(self, *args, **kwargs):
#         encoded: torch.Tensor = self.encoder(*args, **kwargs)  # type: ignore
#         encoded = encoded.mean(dim=1)
#         out: torch.Tensor = self.drop(encoded)
#         out = self.clf(out)

#         return out


class TransformerSequenceClassifier(Classifier):
    def __init__(
        self,
        num_classes,
        num_layers=6,
        hidden_size=512,
        num_heads=8,
        max_length=512,
        inner_size=2048,
        dropout=0.1,
        nystrom=False,
        num_landmarks=32,
        kernel_size=None,
        prenorm=True,
        scalenorm=True,
    ):
        encoder = TransformerSequenceEncoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            max_length=max_length,
            inner_size=inner_size,
            dropout=dropout,
            nystrom=nystrom,
            num_landmarks=num_landmarks,
            kernel_size=kernel_size,
            prenorm=prenorm,
            scalenorm=scalenorm,
        )

        super(TransformerSequenceClassifier, self).__init__(
            encoder, hidden_size, num_classes, dropout=dropout
        )


class TransformerTokenSequenceClassifier(Classifier):
    def __init__(
        self,
        num_classes,
        vocab_size=30000,
        num_layers=6,
        hidden_size=512,
        num_heads=8,
        max_length=512,
        inner_size=2048,
        dropout=0.1,
        nystrom=False,
        num_landmarks=32,
        kernel_size=None,
        prenorm=True,
        scalenorm=True,
    ):
        encoder = TransformerTokenSequenceEncoder(
            vocab_size=vocab_size,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            max_length=max_length,
            inner_size=inner_size,
            dropout=dropout,
            nystrom=nystrom,
            num_landmarks=num_landmarks,
            kernel_size=kernel_size,
            prenorm=prenorm,
            scalenorm=scalenorm,
        )

        super(TransformerTokenSequenceClassifier, self).__init__(
            encoder, hidden_size, num_classes, dropout=dropout
        )


class RNNSequenceClassifier(Classifier):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 256,
        batch_first: bool = True,
        layers: int = 1,
        bidirectional: bool = False,
        merge_bi: str = "cat",
        dropout: float = 0.1,
        rnn_type: str = "lstm",
        packed_sequence: bool = True,
        attention: bool = False,
        max_length: int = -1,
        num_heads: int = 1,
        nystrom: bool = True,
        num_landmarks: int = 32,
        kernel_size: Optional[int] = 33,
        inverse_iterations: int = 6,
    ):
        encoder = AttentiveRNN(
            input_size,
            hidden_size=hidden_size,
            batch_first=batch_first,
            layers=layers,
            bidirectional=bidirectional,
            merge_bi=merge_bi,
            dropout=dropout,
            rnn_type=rnn_type,
            packed_sequence=packed_sequence,
            attention=attention,
            max_length=max_length,
            num_heads=num_heads,
            nystrom=nystrom,
            num_landmarks=num_landmarks,
            kernel_size=kernel_size,
            inverse_iterations=inverse_iterations,
        )

        super(RNNSequenceClassifier, self).__init__(
            encoder, encoder.out_size, num_classes, dropout=dropout
        )


class RNNTokenSequenceClassifier(Classifier):
    def __init__(
        self,
        num_classes: int,
        vocab_size: Optional[int] = None,
        embeddings_dim: Optional[int] = None,
        embeddings: Optional[np.ndarray] = None,
        embeddings_dropout: float = 0.0,
        finetune_embeddings: bool = False,
        hidden_size: int = 256,
        batch_first: bool = True,
        layers: int = 1,
        bidirectional: bool = False,
        merge_bi: str = "cat",
        dropout: float = 0.1,
        rnn_type: str = "lstm",
        packed_sequence: bool = True,
        attention: bool = False,
        max_length: int = -1,
        num_heads: int = 1,
        nystrom: bool = True,
        num_landmarks: int = 32,
        kernel_size: Optional[int] = 33,
        inverse_iterations: int = 6,
    ):
        encoder = TokenRNN(
            vocab_size=vocab_size,
            embeddings_dim=embeddings_dim,
            embeddings=embeddings,
            embeddings_dropout=embeddings_dropout,
            finetune_embeddings=finetune_embeddings,
            hidden_size=hidden_size,
            batch_first=batch_first,
            layers=layers,
            bidirectional=bidirectional,
            merge_bi=merge_bi,
            dropout=dropout,
            rnn_type=rnn_type,
            packed_sequence=packed_sequence,
            attention=attention,
            max_length=max_length,
            num_heads=num_heads,
            nystrom=nystrom,
            num_landmarks=num_landmarks,
            kernel_size=kernel_size,
            inverse_iterations=inverse_iterations,
        )

        super(RNNTokenSequenceClassifier, self).__init__(
            encoder, encoder.out_size, num_classes, dropout=dropout
        )


class TransformerLateFusionClassifier(nn.Module):
    def __init__(
        self,
        modality_feature_sizes,
        num_classes,
        num_layers=2,
        hidden_size=100,
        num_heads=4,
        max_length=512,
        inner_size=400,
        dropout=0.1,
        nystrom=True,
        num_landmarks=32,
        kernel_size=33,
        prenorm=True,
        scalenorm=True,
    ):
        super(TransformerLateFusionClassifier, self).__init__()
        self.modalities = modality_feature_sizes.keys()
        self.modality_encoders = nn.ModuleDict(
            {
                m: TransformerSequenceEncoder(
                    modality_feature_sizes[m],
                    feature_normalization=True if m == "audio" else False,
                    num_layers=num_layers,
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    max_length=max_length,
                    inner_size=inner_size,
                    dropout=dropout,
                    nystrom=nystrom,
                    num_landmarks=num_landmarks,
                    kernel_size=kernel_size,
                    prenorm=prenorm,
                    scalenorm=scalenorm,
                )
                for m in self.modalities
            }
        )
        self.out_size = sum([e.out_size for e in self.modality_encoders.values()])
        self.clf = nn.Linear(self.out_size, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs, attention_masks=None):
        if attention_masks is None:
            attention_masks = dict(
                zip(self.modalities, [None for _ in self.modalities])
            )

        encoded = [
            self.modality_encoders[m](inputs[m], attention_mask=attention_masks[m])
            for m in self.modalities
        ]

        fused = torch.cat(encoded, dim=-1)
        fused = self.drop(fused)
        out = self.clf(fused)

        return out


class RNNLateFusionClassifier(nn.Module):
    def __init__(
        self,
        modality_feature_sizes,
        num_classes,
        num_layers=2,
        batch_first=True,
        bidirectional=True,
        packed_sequence=True,
        merge_bi="cat",
        rnn_type="lstm",
        attention=True,
        hidden_size=100,
        num_heads=4,
        max_length=-1,
        dropout=0.1,
        nystrom=True,
        num_landmarks=32,
        kernel_size=33,
    ):
        super(RNNLateFusionClassifier, self).__init__()
        self.modalities = modality_feature_sizes.keys()
        self.modality_encoders = nn.ModuleDict(
            {
                m: AttentiveRNN(
                    modality_feature_sizes[m],
                    hidden_size=hidden_size,
                    batch_first=batch_first,
                    layers=num_layers,
                    bidirectional=bidirectional,
                    merge_bi=merge_bi,
                    dropout=dropout,
                    rnn_type=rnn_type,
                    packed_sequence=packed_sequence,
                    attention=attention,
                    max_length=max_length,
                    num_heads=num_heads,
                    nystrom=nystrom,
                    num_landmarks=num_landmarks,
                    kernel_size=kernel_size,
                )
                for m in self.modalities
            }
        )
        self.out_size = sum([e.out_size for e in self.modality_encoders.values()])
        self.clf = nn.Linear(self.out_size, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs, lengths):
        encoded = [
            self.modality_encoders[m](inputs[m], lengths[m]) for m in self.modalities
        ]

        fused = torch.cat(encoded, dim=-1)
        fused = self.drop(fused)
        out = self.clf(fused)

        return out


class MOSEITextClassifier(RNNSequenceClassifier):
    def forward(self, x, lengths):
        x = x["text"]
        lengths = lengths["text"]
        return super().forward(x, lengths)
