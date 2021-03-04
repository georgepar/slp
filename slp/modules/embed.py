import math
import numpy as np
import torch
import torch.nn as nn

from typing import Optional
from loguru import logger

from slp.modules.regularization import GaussianNoise


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int = 512, max_len: int = 5000):
        """Inject some information about the relative or absolute position of the tokens in the sequence.

        The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.

        PE for even positions:

        $$\\text{PosEncoder}(pos, 2i) = sin(\\frac{pos}{10000^{\\frac{2i}{d}}})$$

        PE for odd positions:

        $$\\text{PosEncoder}(pos, 2i+1) = cos(\\frac{pos}{10000^{\\frac{2i}{d}}})$$

        where $pos$ is the word position and $i$ is the embedding idx

        Implementation modified from pytorch/examples/word_language_model.py

        Args:
            embedding_dim (int): Embedding / model dimension. Defaults to 512.
            max_len (int): Maximum sequence length that can be encoded. Defaults to 5000.
        """
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(10000.0) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate positional embeddings for input and add them to input tensor

        $$out = x + PosEmbed(x)$$

        x is assumed to be batch first

        Args:
            x (torch.Tensor): [B, L, D] input embeddings

        Returns:
            torch.Tensor: Embeddings + positional embeddings
        """
        x = x + self.pe[:, : x.size(1), :]  # type: ignore
        return x


class Embed(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        embeddings: Optional[np.ndarray] = None,
        noise: float = 0.0,
        dropout: float = 0.0,
        scale: float = 1.0,
        trainable: bool = False,
    ):
        """
        Define the layer of the model and perform the initializations
        of the layers (wherever it is necessary)

        Args:
            num_embeddings (int): Total number of embeddings.
            embeddings_dim (int): Embedding dimension.
            embeddings (numpy.ndarray): the 2D ndarray with the word vectors.
            noise (float): Optional additive noise. Defaults to 0.0.
            dropout (float): Embedding dropout probability. Defaults to 0.0.
            scale (float): Scale word embeddings by a constant. Defaults to 1.0.
            trainable (bool): Finetune embeddings. Defaults to False
        """
        super(Embed, self).__init__()
        self.scale = scale  # scale embeddings by value. Needed for transformer
        # define the embedding layer, with the corresponding dimensions
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        )

        if embeddings is not None:
            logger.info("Initializing Embedding layer with pre-trained weights.")
            if trainable:
                logger.info("Embeddings are going to be finetuned")
            else:
                logger.info("Embeddings are frozen")
            self.init_embeddings(embeddings, trainable)

        # the dropout "layer" for the word embeddings
        self.dropout = nn.Dropout(dropout)

        # the gaussian noise "layer" for the word embeddings
        self.noise = GaussianNoise(noise)

    def init_embeddings(self, weights: np.ndarray, trainable: bool):
        """Initialize embeddings matrix with pretrained embeddings

        Args:
            weights (np.ndarray): pretrained embeddings
            trainable (bool): Finetune embeddings?
        """
        self.embedding.weight = nn.Parameter(
            torch.from_numpy(weights), requires_grad=trainable
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed input tokens

        Assign embedding that corresponds to each token.
        Optionally add Gaussian noise and embedding dropout and scale embeddings by a constant.

        Args:
            x (torch.Tensor): [B, L] Input token ids.

        Returns:
            (torch.Tensor) -> [B, L, E] Embedded tokens.
        """
        embeddings = self.embedding(x)

        if self.noise.stddev > 0:
            embeddings = self.noise(embeddings)

        if self.dropout.p > 0:
            embeddings = self.dropout(embeddings)

        return embeddings * self.scale  # type: ignore
