import torch
import torch.nn as nn


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
        out: torch.Tensor = self.drop(out)
        out = self.clf(out)
        return out
