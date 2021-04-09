import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from slp.util.pytorch import moore_penrose_pinv


def reset_parameters(named_parameters):
    """Initialize parameters in the transformer model."""

    for name, p in named_parameters:
        if "weight" in name:
            nn.init.xavier_normal_(p)

        if "bias" in name:
            nn.init.constant_(p, 0.0)


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Split input tensor into multiple attention heads

    (Batch size, Length, Attention size) => (Batch size, Heads, Lengths, Attention size / Heads)

    Args:
        x (torch.Tensor): [B, L, A] input tensor
        num_heads (int): number of heads

    Returns:
        torch.Tensor: [B, H, L, A/H] Splitted / reshaped tensor
    """
    batch_size, max_length, attention_size = x.size()
    head_size = int(attention_size / num_heads)

    return x.view(batch_size, max_length, num_heads, head_size).permute(0, 2, 1, 3)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
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


def attention_scores(
    k: torch.Tensor,
    q: torch.Tensor,
    dk: int,
    attention_mask: Optional[torch.Tensor] = None,
    dropout: float = 0.2,
    training: bool = True,
) -> torch.Tensor:
    r"""Calculate attention scores for scaled dot product attention

    $$s = softmax(\frac{Q \cdot K^T}{\sqrt{d}})$$

    * B: Batch size
    * L: Keys Sequence length
    * M: Queries Sequence length
    * H: Number of heads
    * A: Feature dimension

    Args:
        k (torch.Tensor): Single head [B, L, A] or multi-head [B, H, L, A/H] Keys tensor
        q (torch.Tensor): Single head [B, M, A] or multi-head [B, H, M, A/H] Keys tensor
        dk (int): Model dimension
        attention_mask (Optional[torch.Tensor]): Optional [B, [H], 1, L] pad mask or [B, [H], M, L] pad mask + subsequent mask
            tensor with zeros in sequence indices that should be masked and ones in sequence indices that should be
            preserved. Defaults to None.
        dropout (float): Drop probability. Defaults to 0.2.
        training (bool): Is module in training phase? Defaults to True.

    Returns:
        torch.Tensor: [B, M, L] or [B, H, M, L] attention scores
    """
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dk)

    if attention_mask is not None:
        scores = scores + ((1 - attention_mask) * -1e5)
    scores = F.softmax(scores, dim=-1)
    scores = F.dropout(scores, p=dropout, training=training)

    return scores


def attention(
    k: torch.Tensor,
    q: torch.Tensor,
    v: torch.Tensor,
    dk: int,
    attention_mask: Optional[torch.Tensor] = None,
    dropout: float = 0.2,
    training: bool = True,
):
    r"""Reweight values using scaled dot product attention

    $$s = softmax(\frac{Q \cdot K^T}{\sqrt{d}}) V$$

    * B: Batch size
    * L: Keys Sequence length
    * M: Queries Sequence length
    * H: Number of heads
    * A: Feature dimension

    Args:
        k (torch.Tensor): Single head [B, L, A] or multi-head [B, H, L, A/H] Keys tensor
        q (torch.Tensor): Single head [B, M, A] or multi-head [B, H, M, A/H] Keys tensor
        v (torch.Tensor): Single head [B, M, A] or multi-head [B, H, M, A/H] Values tensor
        dk (int): Model dimension
        attention_mask (Optional[torch.Tensor]): Optional [B, [H], 1, L] pad mask or [B, [H], M, L] pad mask + subsequent mask
            tensor with zeros in sequence indices that should be masked and ones in sequence indices that should be
            preserved. Defaults to None.
        dropout (float): Drop probability. Defaults to 0.2.
        training (bool): Is module in training phase? Defaults to True.

    Returns:
        torch.Tensor: [B, M, L] or [B, H, M, L] attention scores
    """

    scores = attention_scores(
        k, q, dk, attention_mask=attention_mask, dropout=dropout, training=training
    )
    out = torch.matmul(scores, v)

    return out, scores


def pad_for_nystrom(
    x: torch.Tensor, num_landmarks: int, attention_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Pad inputs and attention_mask to perform Nystrom Attention

    Pad to nearest multiple of num_landmarks

    Args:
        x (torch.Tensor): [B, L, A] Input tensor
        num_landmarks (int): Number of landmark points
        attention_mask (Optional[torch.Tensor]): [B, L] Padding mask

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]: Padded inputs and attention_mask
    """
    if attention_mask is not None:
        attention_mask = attention_mask.squeeze()

    _, seq_length, _ = x.size()

    _, remainder = (
        math.ceil(seq_length / num_landmarks),
        seq_length % num_landmarks,
    )

    if remainder > 0:
        padding = num_landmarks - remainder
        x = F.pad(x, (0, 0, padding, 0), value=0)

        if attention_mask is not None:
            attention_mask = F.pad(attention_mask, (padding, 0))

    return x, attention_mask


def nystrom_attention(
    k: torch.Tensor,
    q: torch.Tensor,
    v: torch.Tensor,
    dk: int,
    num_landmarks: int,
    attention_mask: Optional[torch.Tensor] = None,
    inverse_iterations: int = 6,
    dropout: float = 0.2,
    training: bool = True,
):
    """Calculate attention using nystrom approximation

    Implementation heavily based on: https://github.com/lucidrains/nystrom-attention

    Reference: https://arxiv.org/abs/2102.03902
    * B: Batch size
    * L: Keys Sequence length
    * M: Queries Sequence length
    * H: Number of heads
    * A: Feature dimension

    Args:
        k (torch.Tensor): Single head [B, L, A] or multi-head [B, H, L, A/H] Keys tensor
        q (torch.Tensor): Single head [B, M, A] or multi-head [B, H, M, A/H] Keys tensor
        v (torch.Tensor): Single head [B, M, A] or multi-head [B, H, M, A/H] Values tensor
        dk (int): Model dimension
        num_landmarks (int): Number of landmark points
        attention_mask (Optional[torch.Tensor]): Optional [B, [H], 1, L] pad mask or [B, [H], M, L] pad mask + subsequent mask
            tensor with zeros in sequence indices that should be masked and ones in sequence indices that should be
            preserved. Defaults to None.
        inverse_iterations (int): Number of iterations for Moore Penrose iterative inverse
            approximation
        dropout (float): Drop probability. Defaults to 0.2.
        training (bool): Is module in training phase? Defaults to True.

    Returns:
        torch.Tensor: [B, M, L] or [B, H, M, L] attention scores
    """
    _, num_heads, seq_length, head_size = k.size()

    masked_mean_denom = seq_length // num_landmarks
    if attention_mask is not None:
        attention_mask = attention_mask.unsqueeze(1)
        masked_mean_denom = (
            attention_mask.reshape(-1, 1, num_landmarks, seq_length // num_landmarks).sum(-1) + 1e-8  # type: ignore
        )  # (B, 1, Landmarks)
        mask_landmarks = (masked_mean_denom > 0).type(torch.float)  # type: ignore
        masked_mean_denom = masked_mean_denom[..., None]  # type: ignore
        attention_mask = attention_mask.unsqueeze(-1)
        q = q * attention_mask  # (B, H, L, A/H)
        k = k * attention_mask  # (B, H, L, A/H)
        v = v * attention_mask  # (B, H, L, A/H)

        scores_1_mask = attention_mask * mask_landmarks[..., None, :]
        scores_2_mask = mask_landmarks[..., None] * mask_landmarks[..., None, :]
        scores_3_mask = scores_1_mask.transpose(-1, -2)

    q = q / math.sqrt(dk)

    q_landmarks = q.reshape(
        q.size(0),  # batch_size
        q.size(1),  # num_heads
        num_landmarks,  # landmarks
        seq_length // num_landmarks,  # reduced length
        q.size(-1),  # head_size
    ).sum(
        dim=-2
    )  # (B, H, Landmarks, A/H)

    k_landmarks = k.reshape(
        k.size(0),  # batch_size
        k.size(1),  # num_heads
        num_landmarks,  # landmarks
        seq_length // num_landmarks,  # reduced length
        k.size(-1),  # head size
    ).sum(
        dim=-2
    )  # (B, H, Landmarks, A/H)

    k_landmarks = k_landmarks / masked_mean_denom
    q_landmarks = q_landmarks / masked_mean_denom

    scores_1 = attention_scores(
        k_landmarks,
        q,
        1,  # We have already accounted for dk
        attention_mask=scores_1_mask,
        dropout=dropout,
        training=training,
    )

    scores_2 = attention_scores(
        k_landmarks,
        q_landmarks,
        1,  # We have already accounted for dk
        attention_mask=scores_2_mask,
        dropout=dropout,
        training=training,
    )

    scores_3 = attention_scores(
        k,
        q_landmarks,
        1,  # We have already accounted for dk
        attention_mask=scores_3_mask,
        dropout=dropout,
        training=training,
    )

    z_star = moore_penrose_pinv(scores_2, num_iter=inverse_iterations)
    out = (scores_1 @ z_star) @ (scores_3 @ v)

    return out, (scores_1, scores_2, scores_3)


class SelfAttention(nn.Module):
    def __init__(
        self,
        attention_size: int = 512,
        input_size: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """Single-Headed Dot-product self attention module

        Args:
            attention_size (int): Number of hidden features. Defaults to 512.
            input_size (Optional[int]): Input features. Defaults to None.
                If None input_size is set to attention_size.
            dropout (float): Drop probability. Defaults to 0.1.
        """
        super(SelfAttention, self).__init__()

        if input_size is None:
            input_size = attention_size
        self.dk = input_size
        self.kqv = nn.Linear(input_size, 3 * attention_size, bias=False)
        self.dropout = dropout
        reset_parameters(self.named_parameters())

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Single-head scaled dot-product attention forward pass

        Outputs the values, where features for each sequence element are weighted by their respective attention scores

        $$a = softmax(\frac{Q}{K^T}){\sqrt{d}}) \dot V$$

        * B: Batch size
        * L: Keys Sequence length
        * M: Queries Sequence length
        * H: Number of heads
        * A: Feature dimension

        Args:
            x (torch.Tensor): [B, L, D] Input tensor
            attention_mask (Optional[torch.Tensor]): Optional [B, L] or [B, M, L] zero-one mask for sequence elements. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (Reweighted values [B, L, D], attention scores [B, M, L])
        """
        if attention_mask is not None:
            if len(list(attention_mask.size())) == 2:
                attention_mask = attention_mask.unsqueeze(1)

        k, q, v = self.kqv(x).chunk(3, dim=-1)  # (B, L, A)

        # weights => (B, L, L)
        out, scores = attention(
            k,
            q,
            v,
            self.dk,
            attention_mask=attention_mask,
            dropout=self.dropout,
            training=self.training,
        )

        return out, scores


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
        reset_parameters(self.named_parameters())

    def forward(
        self,
        keys: torch.Tensor,
        queries: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Single-head scaled dot-product attention forward pass

        Outputs the values, where features for each sequence element are weighted by their respective attention scores

        $$a = softmax(\frac{Q}{K^T}){\sqrt{d}}) \dot V$$

        * B: Batch size
        * L: Keys Sequence length
        * M: Queries Sequence length
        * H: Number of heads
        * A: Feature dimension

        Args:
            keys (torch.Tensor): [B, L, D] Keys tensor
            queries (Optional[torch.Tensor]): Optional [B, M, D] Queries tensor. If None queries = keys. Defaults to None.
            attention_mask (Optional[torch.Tensor]): Optional [B, L] or [B, M, L] zero-one mask for sequence elements. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (Reweighted values [B, L, D], attention scores [B, M, L])
        """
        if attention_mask is not None:
            if len(list(attention_mask.size())) == 2:
                attention_mask = attention_mask.unsqueeze(1)

        if queries is None:
            queries = keys

        values = keys

        k = self.k(keys)  # (B, L, A)
        q = self.q(queries)
        v = self.v(values)

        # weights => (B, L, L)
        out, scores = attention(
            k,
            q,
            v,
            self.dk,
            attention_mask=attention_mask,
            dropout=self.dropout,
            training=self.training,
        )

        return out, scores


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        attention_size: int = 512,
        num_heads: int = 8,
        input_size: Optional[int] = None,
        dropout: float = 0.1,
        nystrom: bool = False,
        num_landmarks: int = 64,
        inverse_iterations: int = 6,
        kernel_size: Optional[int] = None,
    ):
        """Multi-Headed Dot-product attention module

        Args:
            attention_size (int): Number of hidden features. Defaults to 512.
            num_heads (int): Number of attention heads
            input_size (Optional[int]): Input features. Defaults to None.
                If None input_size is set to attention_size.
            dropout (float): Drop probability. Defaults to 0.1.
        """
        super(MultiheadSelfAttention, self).__init__()

        if input_size is None:
            input_size = attention_size
        self.inverse_iterations = inverse_iterations
        self.num_landmarks = num_landmarks
        self.nystrom = nystrom
        self.num_heads = num_heads
        self.head_size = int(attention_size / num_heads)
        self.dk = self.head_size
        self.attention_size = attention_size
        self.kqv = nn.Linear(input_size, 3 * attention_size, bias=False)
        self.output = nn.Linear(attention_size, attention_size)
        self.dropout = dropout

        self.conv = None

        if kernel_size is not None:
            self.conv = nn.Conv2d(
                in_channels=self.num_heads,
                out_channels=self.num_heads,
                kernel_size=(kernel_size, 1),
                padding=(kernel_size // 2, 0),
                bias=False,
                groups=self.num_heads,
            )

        reset_parameters(self.named_parameters())

    def forward(self, x, attention_mask=None):
        r"""Multi-head scaled dot-product attention forward pass

        Outputs the values, where features for each sequence element are weighted by their respective attention scores

        Each head performs dot-product attention

        $$a_H = softmax(\frac{Q_H \cdot K_H^T}{\sqrt{d}}) \cdot V_H$$

        The outputs of multiple heads are concatenated and passed through a feedforward layer.

        $$a = W (a^{(1)}_{H} \mathbin\Vert a^{(2)}_{H} \dots) + b$$


        * B: Batch size
        * L: Keys Sequence length
        * M: Queries Sequence length
        * H: Number of heads
        * A: Feature dimension


        Args:
            x (torch.Tensor): [B, L, D] Keys tensor
            attention_mask (Optional[torch.Tensor]): Optional [B, M, L] zero-one mask for sequence elements. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (Reweighted values [B, L, D], attention scores [B, H, M, L])
        """
        _, seq_length, _ = x.size()

        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.unsqueeze(1)

        if self.nystrom:
            x, attention_mask = pad_for_nystrom(
                x, self.num_landmarks, attention_mask=attention_mask
            )

        k, q, v = self.kqv(x).chunk(3, dim=-1)
        k = split_heads(k, self.num_heads)
        q = split_heads(q, self.num_heads)
        v = split_heads(v, self.num_heads)

        if self.nystrom:
            # out = (B, H, L, A/H)
            # scores = Tuple
            out, scores = nystrom_attention(
                k,
                q,
                v,
                self.dk,
                self.num_landmarks,
                attention_mask=attention_mask,
                inverse_iterations=self.inverse_iterations,
                dropout=self.dropout,
                training=self.training,
            )
        else:
            # out => (B, H, L, A/H)
            # scores => (B, H, L, L)
            out, scores = attention(
                k,
                q,
                v,
                self.dk,
                attention_mask=attention_mask,
                dropout=self.dropout,
                training=self.training,
            )

        if self.conv is not None:
            if attention_mask is None or attention_mask.ndim > 2:
                out = out + self.conv(v)
            else:
                attention_mask = attention_mask.squeeze()
                out = out + self.conv(v * attention_mask[:, None, :, None])

        # out => (B, H, L, A/H)
        out = merge_heads(out)
        if out.size(1) != seq_length:
            out = out[:, -seq_length:, :]
        out = self.output(out)

        return out, scores


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        attention_size: int = 512,
        num_heads: int = 8,
        input_size: Optional[int] = None,
        dropout: float = 0.1,
        nystrom: bool = False,
        num_landmarks: int = 64,
        inverse_iterations: int = 6,
        kernel_size: Optional[int] = None,
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
        self.inverse_iterations = inverse_iterations
        self.num_landmarks = num_landmarks
        self.nystrom = nystrom
        self.num_heads = num_heads
        self.head_size = int(attention_size / num_heads)
        self.dk = self.head_size
        self.attention_size = attention_size
        self.k = nn.Linear(input_size, attention_size, bias=False)
        self.q = nn.Linear(input_size, attention_size, bias=False)
        self.v = nn.Linear(input_size, attention_size, bias=False)
        self.output = nn.Linear(attention_size, attention_size)
        self.dropout = dropout

        self.conv = None

        if kernel_size is not None:
            self.conv = nn.Conv2d(
                in_channels=self.num_heads,
                out_channels=self.num_heads,
                kernel_size=(kernel_size, 1),
                padding=(kernel_size // 2, 0),
                bias=False,
                groups=self.num_heads,
            )

        reset_parameters(self.named_parameters())

    def forward(self, keys, queries=None, attention_mask=None):
        r"""Multi-head scaled dot-product attention forward pass

        Outputs the values, where features for each sequence element are weighted by their respective attention scores

        Each head performs dot-product attention

        $$a_H = softmax(\frac{Q_H \cdot K_H^T}{\sqrt{d}}) \cdot V_H$$

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
            attention_mask (Optional[torch.Tensor]): Optional [B, M, L] zero-one mask for sequence elements. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (Reweighted values [B, L, D], attention scores [B, H, M, L])
        """
        _, seq_length, _ = keys.size()

        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.unsqueeze(1)

        if self.nystrom:
            keys, attention_mask = pad_for_nystrom(
                keys, self.num_landmarks, attention_mask=attention_mask
            )

        if queries is None:
            queries = keys

        values = keys

        k = self.k(keys)
        q = self.q(queries)
        v = self.v(values)
        k = split_heads(k, self.num_heads)
        q = split_heads(q, self.num_heads)
        v = split_heads(v, self.num_heads)

        if self.nystrom:
            # out = (B, H, L, A/H)
            # scores = Tuple
            out, scores = nystrom_attention(
                k,
                q,
                v,
                self.dk,
                self.num_landmarks,
                attention_mask=attention_mask,
                inverse_iterations=self.inverse_iterations,
                dropout=self.dropout,
                training=self.training,
            )
        else:
            # out => (B, H, L, A/H)
            # scores => (B, H, L, L)
            out, scores = attention(
                k,
                q,
                v,
                self.dk,
                attention_mask=attention_mask,
                dropout=self.dropout,
                training=self.training,
            )

        if self.conv is not None:
            if attention_mask is None or attention_mask.ndim > 2:
                out += self.conv(v)
            else:
                attention_mask = attention_mask.squeeze()
                out += self.conv(v * attention_mask[:, None, :, None])

        # out => (B, H, L, A/H)
        out = merge_heads(out)
        if out.size(1) != seq_length:
            out = out[:, :seq_length, :]
        out = self.output(out)

        return out, scores
