from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from slp.modules.attention import MultiheadSelfAttention as MultiheadAttention
from slp.modules.attention import SelfAttention as Attention
from slp.modules.embed import Embed
from slp.util.pytorch import PackSequence, PadPackedSequence, pad_mask

# from slp.modules.attention import (
#     Attention,
#     MultiheadAttention,
# )


class RNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        batch_first: bool = True,
        layers: int = 1,
        bidirectional: bool = False,
        merge_bi: str = "cat",
        dropout: float = 0.0,
        rnn_type: str = "lstm",
        packed_sequence: bool = True,
        max_length: int = -1,
    ):
        """LSTM - GRU wrapper with packed sequence support and handling for bidirectional / last output states

        It is recommended to run with batch_first=True because the rest of the code is built with this assumption

        Args:
            input_size (int): Input features.
            hidden_size (int): Hidden features.
            batch_first (bool): Use batch first representation type. Defaults to True.
            layers (int): Number of RNN layers. Defaults to 1.
            bidirectional (bool): Use bidirectional RNNs. Defaults to False.
            merge_bi (str): How bidirectional states are merged. Defaults to "cat".
            dropout (float): Dropout probability. Defaults to 0.0.
            rnn_type (str): lstm or gru. Defaults to "lstm".
            packed_sequence (bool): Use packed sequences. Defaults to True.
        """
        super(RNN, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.merge_bi = merge_bi
        self.rnn_type = rnn_type.lower()

        if not batch_first:
            logger.warning(
                "You are running RNN with batch_first=False. Make sure this is really what you want"
            )

        if not packed_sequence:
            logger.warning(
                "You have set packed_sequence=False. Running with packed_sequence=True will be much faster"
            )

        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size,
            hidden_size,
            batch_first=batch_first,
            num_layers=layers,
            bidirectional=bidirectional,
        )
        self.drop = nn.Dropout(dropout)
        self.packed_sequence = packed_sequence

        if packed_sequence:
            self.pack = PackSequence(batch_first=batch_first)
            self.unpack = PadPackedSequence(
                batch_first=batch_first, max_length=max_length
            )

    @property
    def out_size(cls) -> int:
        """RNN output features size

        Returns:
            int: RNN output features size
        """
        out: int = (
            2 * cls.hidden_size
            if cls.bidirectional and cls.merge_bi == "cat"
            else cls.hidden_size
        )

        return out

    def _merge_bi(self, forward: torch.Tensor, backward: torch.Tensor) -> torch.Tensor:
        """Merge forward and backward states

        Args:
            forward (torch.Tensor): [B, L, H] Forward states
            backward (torch.Tensor): [B, L, H] Backward states

        Returns:
            torch.Tensor: [B, L, H] or [B, L, 2*H] Merged forward and backward states
        """

        if self.merge_bi == "sum":
            return forward + backward

        return torch.cat((forward, backward), dim=-1)

    def _select_last_unpadded(
        self, out: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Get the last timestep before padding starts

        Args:
            out (torch.Tensor): [B, L, H] Fprward states
            lengths (torch.Tensor): [B] Original sequence lengths

        Returns:
            torch.Tensor: [B, H] Features for last sequence timestep
        """
        gather_dim = 1 if self.batch_first else 0
        gather_idx = (
            (lengths - 1)  # -1 to convert to indices
            .unsqueeze(1)  # (B) -> (B, 1)
            .expand((-1, self.hidden_size))  # (B, 1) -> (B, H)
            # (B, 1, H) if batch_first else (1, B, H)
            .unsqueeze(gather_dim)
        )
        # Last forward for real length or seq (unpadded tokens)
        last_out = out.gather(gather_dim, gather_idx).squeeze(gather_dim)

        return last_out

    def _final_output(
        self, out: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create RNN ouputs

        Collect last hidden state for forward and backward states
        Code adapted from https://stackoverflow.com/a/50950188

        Args:
            out (torch.Tensor): [B, L, num_directions * H] RNN outputs
            lengths (torch.Tensor): [B] Original sequence lengths

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (
                merged forward and backward states [B, L, H] or [B, L, 2*H],
                merged last forward and backward state [B, H] or [B, 2*H]
            )
        """

        if not self.bidirectional:
            return out, self._select_last_unpadded(out, lengths)

        forward, backward = (out[..., : self.hidden_size], out[..., self.hidden_size :])
        # Last backward corresponds to first token
        last_backward_out = backward[:, 0, :] if self.batch_first else backward[0, ...]
        # Last forward for real length or seq (unpadded tokens)
        last_forward_out = self._select_last_unpadded(forward, lengths)
        out = self._merge_bi(forward, backward) if self.merge_bi != "cat" else out

        return out, self._merge_bi(last_forward_out, last_backward_out)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ]:
        """RNN forward pass

        Args:
            x (torch.Tensor): [B, L, D] Input features
            lengths (torch.Tensor): [B] Original sequence lengths

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (
                merged forward and backward states [B, L, H] or [B, L, 2*H],
                merged last forward and backward state [B, H] or [B, 2*H],
                hidden states tuple of [num_layers * num_directions, B, H] for LSTM or tensor [num_layers * num_directions, B, H] for GRU
            )
        """
        self.rnn.flatten_parameters()

        if self.packed_sequence:
            # Latest pytorch allows only cpu tensors for packed sequence
            lengths = lengths.to("cpu")
            x, lengths = self.pack(x, lengths)
        out, hidden = self.rnn(x)

        if self.packed_sequence:
            out = self.unpack(out, lengths)
        out = self.drop(out)
        lengths = lengths.to(out.device)

        out, last_timestep = self._final_output(out, lengths)

        return out, last_timestep, hidden


class AttentiveRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
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
        """RNN with embedding layer and optional attention mechanism

        Single-headed scaled dot-product attention is used as an attention mechanism

        Args:
            input_size (int): Input features dimension
            hidden_size (int): Hidden features
            batch_first (bool): Use batch first representation type. Defaults to True.
            layers (int): Number of RNN layers. Defaults to 1.
            bidirectional (bool): Use bidirectional RNNs. Defaults to False.
            merge_bi (str): How bidirectional states are merged. Defaults to "cat".
            dropout (float): Dropout probability. Defaults to 0.0.
            rnn_type (str): lstm or gru. Defaults to "lstm".
            packed_sequence (bool): Use packed sequences. Defaults to True.
            max_length (int): Maximum sequence length for fixed length padding. If -1 takes the
                largest sequence length in this batch
            attention (bool): Use attention mechanism. Defaults to False
            num_heads (int): Number of attention heads. If 1 uses single headed attention
            nystrom (bool): Use nystrom approximation for multihead attention
            num_landmarks (int): Number of landmark sequence elements for nystrom attention
            kernel_size (int): Kernel size for multihead attention output residual convolution
            inverse_iterations (int): Number of iterations for moore-penrose inverse approximation
                in nystrom attention. 6 is a good value
        """
        super(AttentiveRNN, self).__init__()
        self.rnn = RNN(
            input_size,  # type: ignore
            hidden_size,
            batch_first=batch_first,
            layers=layers,
            merge_bi=merge_bi,
            bidirectional=bidirectional,
            dropout=dropout,
            rnn_type=rnn_type,
            packed_sequence=packed_sequence,
            max_length=max_length,
        )
        self.out_size = (
            hidden_size
            if not (bidirectional and merge_bi == "cat")
            else 2 * hidden_size
        )
        self.batch_first = batch_first

        self.attention = None

        if attention:
            if num_heads == 1:
                self.attention = Attention(
                    attention_size=self.out_size, dropout=dropout
                )
            else:
                self.attention = MultiheadAttention(  # type: ignore
                    attention_size=self.out_size,
                    num_heads=num_heads,
                    kernel_size=kernel_size,
                    nystrom=nystrom,
                    num_landmarks=num_landmarks,
                    inverse_iterations=inverse_iterations,
                    dropout=dropout,
                )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Attentive RNN forward pass

        If self.attention=True then the outputs are the weighted sum of the RNN hidden states with the attention score weights
        Else the output is the last hidden state of the RNN.

        Args:
            x (torch.Tensor): [B, L] Input token ids
            lengths (torch.Tensor): [B] Original sequence lengths

        Returns:
            torch.Tensor: [B, H] or [B, 2*H] Output features to be used for classification
        """
        out, last_hidden, _ = self.rnn(x, lengths)

        if self.attention is not None:
            out, _ = self.attention(
                out,
                attention_mask=pad_mask(
                    lengths, max_length=out.size(1) if self.batch_first else out.size(0)
                ),
            )
            out = out.mean(dim=1)
        else:
            out = last_hidden

        return out  # type: ignore


class TokenRNN(nn.Module):
    def __init__(
        self,
        hidden_size: int = 256,
        vocab_size: Optional[int] = None,
        embeddings_dim: Optional[int] = None,
        embeddings: Optional[np.ndarray] = None,
        embeddings_dropout: float = 0.0,
        finetune_embeddings: bool = False,
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
        """RNN with embedding layer and optional attention mechanism

        Single-headed scaled dot-product attention is used as an attention mechanism

        Args:
            hidden_size (int): Hidden features
            vocab_size (Optional[int]): Vocabulary size. Defaults to None.
            embeddings_dim (Optional[int]): Embedding dimension. Defaults to None.
            embeddings (Optional[np.ndarray]): Embedding matrix. Defaults to None.
            embeddings_dropout (float): Embedding dropout probability. Defaults to 0.0.
            finetune_embeddings (bool): Finetune embeddings? Defaults to False.
            batch_first (bool): Use batch first representation type. Defaults to True.
            layers (int): Number of RNN layers. Defaults to 1.
            bidirectional (bool): Use bidirectional RNNs. Defaults to False.
            merge_bi (str): How bidirectional states are merged. Defaults to "cat".
            dropout (float): Dropout probability. Defaults to 0.0.
            rnn_type (str): lstm or gru. Defaults to "lstm".
            packed_sequence (bool): Use packed sequences. Defaults to True.
            max_length (int): Maximum sequence length for fixed length padding. If -1 takes the
                largest sequence length in this batch
            attention (bool): Use attention mechanism. Defaults to False
            num_heads (int): Number of attention heads. If 1 uses single headed attention
            nystrom (bool): Use nystrom approximation for multihead attention
            num_landmarks (int): Number of landmark sequence elements for nystrom attention
            kernel_size (int): Kernel size for multihead attention output residual convolution
            inverse_iterations (int): Number of iterations for moore-penrose inverse approximation
                in nystrom attention. 6 is a good value
        """
        super(TokenRNN, self).__init__()

        if embeddings is None:
            finetune_embeddings = True
            assert (
                vocab_size is not None
            ), "You should either pass an embeddings matrix or vocab size"
            assert (
                embeddings_dim is not None
            ), "You should either pass an embeddings matrix or embeddings_dim"
        else:
            vocab_size = embeddings.shape[0]
            embeddings_dim = embeddings.shape[1]

        self.embed = Embed(
            vocab_size,  # type: ignore
            embeddings_dim,  # type: ignore
            embeddings=embeddings,
            dropout=embeddings_dropout,
            scale=hidden_size ** 0.5,
            trainable=finetune_embeddings,
        )
        self.encoder = AttentiveRNN(
            embeddings_dim,  # type: ignore
            hidden_size,
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

        self.out_size = self.encoder.out_size

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Token RNN forward pass

        If self.attention=True then the outputs are the weighted sum of the RNN hidden states with the attention score weights
        Else the output is the last hidden state of the RNN.

        Args:
            x (torch.Tensor): [B, L] Input token ids
            lengths (torch.Tensor): [B] Original sequence lengths

        Returns:
            torch.Tensor: [B, H] or [B, 2*H] Output features to be used for classification
        """
        x = self.embed(x)
        out = self.encoder(x, lengths)

        return out  # type: ignore
