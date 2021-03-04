import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple

from torch.utils.checkpoint import checkpoint

from slp.modules.feedforward import FF


def attention_scores(
    k: torch.Tensor,
    q: torch.Tensor,
    dk: int,
    attention_mask: Optional[torch.Tensor] = None,
    dropout: float = 0.2,
    training: bool = True,
) -> torch.Tensor:
    """Calculate attention scores for scaled dot product attention

    $$s = softmax(\\frac{Q \cdot K^T}{\sqrt{d}})$$

    * B: Batch size
    * L: Keys Sequence length
    * M: Queries Sequence length
    * H: Number of heads
    * A: Feature dimension

    Args:
        k (torch.Tensor): Single head [B, L, A] or multi-head [B, H, L, A/H] Keys tensor
        q (torch.Tensor): Single head [B, M, A] or multi-head [B, H, M, A/H] Keys tensor
        dk (int): Model dimension
        attention_mask (Optional[torch.Tensor]): Optional [B, M, L] mask tensor with zeros in
            sequence indices that should be masked and ones in sequence indices that should be
            preserved. Defaults to None.
        dropout (float): Drop probability. Defaults to 0.2.
        training (bool): Is module in training phase? Defaults to True.

    Returns:
        torch.Tensor: [B, M, L] or [B, H, M, L] attention scores
    """
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dk)

    if attention_mask is not None:
        scores = scores + ((1 - attention_mask.unsqueeze(1)) * -1e5)
    scores = F.softmax(scores, dim=-1)
    scores = F.dropout(scores, p=dropout, training=training)

    return scores


class Attention(nn.Module):
    def __init__(
        self,
        attention_size: int = 512,
        input_size: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """Single-Headed Dot-product attention module

        Args:
            attention_size (int): Number of hidden features. Defaults to 512.
            input_size (Optional[int]): Input features. Defaults to None.
                If None input_size is set to attention_size.
            dropout (float): Drop probability. Defaults to 0.1.
        """
        super(Attention, self).__init__()

        if input_size is None:
            input_size = attention_size
        self.dk = input_size
        self.k = nn.Linear(input_size, attention_size, bias=False)
        self.q = nn.Linear(input_size, attention_size, bias=False)
        self.v = nn.Linear(input_size, attention_size, bias=False)
        self.dropout = dropout
        self._reset_parameters()

    def forward(
        self,
        keys: torch.Tensor,
        queries: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-head scaled dot-product attention forward pass

        Outputs the values, where features for each sequence element are weighted by their respective attention scores

        $$a = softmax(\\frac{Q}{K^T}){\sqrt{d}}) \dot V$$

        * B: Batch size
        * L: Keys Sequence length
        * M: Queries Sequence length
        * H: Number of heads
        * A: Feature dimension

        Args:
            keys (torch.Tensor): [B, L, D] Keys tensor
            queries (Optional[torch.Tensor]): Optional [B, M, D] Queries tensor. If None queries = keys. Defaults to None.
            values (Optional[torch.Tensor]): Optional [B, L, D] Values tensor. If None values = keys. Defaults to None.
            attention_mask (Optional[torch.Tensor]): Optional [B, M, L] zero-one mask for sequence elements. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (Reweighted values [B, L, D], attention scores [B, M, L])
        """

        if queries is None:
            queries = keys

        if values is None:
            values = keys
        k = self.k(keys)  # (B, L, A)
        q = self.q(queries)  # (B, L, A)
        v = self.v(values)  # (B, L, A)

        # weights => (B, L, L)
        scores = attention_scores(
            k,
            q,
            self.dk,
            attention_mask=attention_mask,
            dropout=self.dropout,
            training=self.training,
        )

        # out => (B, L, A)
        out = torch.bmm(scores, v)

        return out, scores

    def _reset_parameters(self):
        """xavier uniform init for Linear layer weights"""
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.v.weight)


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        attention_size: int = 512,
        num_heads: int = 8,
        input_size: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """Multi-Headed Dot-product attention module

        Args:
            attention_size (int): Number of hidden features. Defaults to 512.
            num_heads (int): Number of attention heads
            input_size (Optional[int]): Input features. Defaults to None.
                If None input_size is set to attention_size.
            dropout (float): Drop probability. Defaults to 0.1.
        """
        super(MultiheadAttention, self).__init__()

        if input_size is None:
            input_size = attention_size
        self.dk = input_size
        self.num_heads = num_heads
        self.head_size = int(attention_size / num_heads)
        self.attention_size = attention_size
        self.k = nn.Linear(input_size, attention_size, bias=False)
        self.q = nn.Linear(input_size, attention_size, bias=False)
        self.v = nn.Linear(input_size, attention_size, bias=False)
        self.output = nn.Linear(attention_size, attention_size)
        self.dropout = dropout
        self._reset_parameters()

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split input tensor into multiple attention heads

        (Batch size, Length, Attention size) => (Batch size, Heads, Lengths, Attention size / Heads)

        Args:
            x (torch.Tensor): [B, L, A] input tensor

        Returns:
            torch.Tensor: [B, H, L, A/H] Splitted / reshaped tensor
        """
        batch_size, max_length, _ = x.size()

        return x.view(batch_size, max_length, self.num_heads, self.head_size).permute(
            0, 2, 1, 3
        )

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merge multiple attention heads into output tensor

        (Batch size, Heads, Lengths, Attention size / Heads) => (Batch size, Length, Attention size)

        Args:
            x (torch.Tensor): [B, H, L, A/H] multi-head tensor

        Returns:
            torch.Tensor:  [B, L, A] merged / reshaped tensor
        """
        batch_size, _, max_length, _ = x.size()
        # x => (B, L, H, A/H)
        x = x.permute(0, 2, 1, 3).contiguous()

        return x.view(batch_size, max_length, -1)

    def forward(self, keys, queries=None, values=None, attention_mask=None):
        """Multi-head scaled dot-product attention forward pass

        Outputs the values, where features for each sequence element are weighted by their respective attention scores

        Each head performs dot-product attention

        $$a_H = softmax(\\frac{Q_H \cdot K_H^T}{\sqrt{d}}) \cdot V_H$$

        The outputs of multiple heads are concatenated and passed through a feedforward layer.

        $$a = W (a^{(1)}_{H} \mathbin\Vert a^{(2)}_{H} \dots) + b$$


        * B: Batch size
        * L: Keys Sequence length
        * M: Queries Sequence length
        * H: Number of heads
        * A: Feature dimension


        Args:
            keys (torch.Tensor): [B, L, D] Keys tensor
            queries (Optional[torch.Tensor]): Optional [B, M, D] Queries tensor. If None queries = keys. Defaults to None.
            values (Optional[torch.Tensor]): Optional [B, L, D] Values tensor. If None values = keys. Defaults to None.
            attention_mask (Optional[torch.Tensor]): Optional [B, M, L] zero-one mask for sequence elements. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (Reweighted values [B, L, D], attention scores [B, H, M, L])
        """
        if queries is None:
            queries = keys

        if values is None:
            values = keys
        k = self._split_heads(self.k(keys))  # (B, H, L, A/H)
        q = self._split_heads(self.q(queries))  # (B, H, L, A/H)
        v = self._split_heads(self.v(values))  # (B, H, L, A/H)

        # scores => (B, H, L, L)
        scores = attention_scores(
            k,
            q,
            self.dk,
            attention_mask=attention_mask,
            dropout=self.dropout,
            training=self.training,
        )

        # out => (B, H, L, A/H)
        out = self._merge_heads(torch.matmul(scores, v))
        out = self.output(out)

        return out

    def _reset_parameters(self):
        """xavier uniform init for Linear layer weights"""
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.v.weight)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.constant_(self.output.bias, 0.0)
