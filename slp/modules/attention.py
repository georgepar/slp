import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from slp.modules.feedforward import FF


def calc_scores(dk):
    def fn(q, k):
        return torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dk)

    return fn


class Attention(nn.Module):
    """Some Information about Attention"""

    def __init__(
        self, attention_size=512, input_size=None, dropout=0.1, grad_checkpoint=False
    ):
        super(Attention, self).__init__()
        if input_size is None:
            input_size = attention_size
        self.dk = input_size
        self.grad_checkpoint = grad_checkpoint
        self.k = nn.Linear(input_size, attention_size, bias=False)
        self.q = nn.Linear(input_size, attention_size, bias=False)
        self.v = nn.Linear(input_size, attention_size, bias=False)
        self.drop = nn.Dropout(dropout)
        self._reset_parameters()

    def forward(self, x, queries=None, values=None, attention_mask=None):
        """
        x : (B, L, D)
        queries : (B, L, D)
        values : (B, L, D)
        """
        if queries is None:
            queries = x
        if values is None:
            values = x
        k = self.k(x)  # (B, L, A)
        q = self.q(queries)  # (B, L, A)
        v = self.v(values)  # (B, L, A)

        # weights => (B, L, L)
        if self.grad_checkpoint:
            scores = checkpoint(calc_scores(self.dk), q, k)
        else:
            scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.dk)
        if attention_mask is not None:
            scores = scores + ((1 - attention_mask.unsqueeze(1)) * -1e5)
        scores = F.softmax(scores, dim=-1)
        scores = self.drop(scores)

        # out => (B, L, A)
        out = torch.bmm(scores, v)
        return out, scores

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.v.weight)


class CoAttention(nn.Module):
    """Some Information about Attention"""

    def __init__(
        self, attention_size=512, input_size=None, query_size=None, dropout=0.1, grad_checkpoint=False
    ):
        super(CoAttention, self).__init__()
        if input_size is None:
            input_size = attention_size

        if query_size is None:
            query_size = attention_size

        self.dk = input_size
        self.grad_checkpoint = grad_checkpoint
        self.k = nn.Linear(input_size, attention_size, bias=False)
        self.q_self = nn.Linear(input_size, attention_size, bias=False)
        self.q_other = nn.Linear(query_size, attention_size, bias=False)
        self.v = nn.Linear(input_size, attention_size, bias=False)
        self.drop = nn.Dropout(dropout)
        self._reset_parameters()

    def forward(self, x, queries, values=None, attention_mask=None):
        """
        x : (B, L, D)
        queries : (B, L, D)
        values : (B, L, D)
        """
        if values is None:
            values = x
        k = self.k(x)  # (B, L, A)
        q_self = self.q_self(x)  # (B, L, A)
        q_other = self.q_other(queries)
        v = self.v(values)  # (B, L, A)

        # weights => (B, L, L)
        if self.grad_checkpoint:
            scores_self = checkpoint(calc_scores(self.dk), q_self, k)
            scores_other = checkpoint(calc_scores(self.dk), q_other, k)
        else:
            scores_self = torch.bmm(q_self, k.transpose(1, 2)) / math.sqrt(self.dk)
            scores_other = torch.bmm(q_other, k.transpose(1, 2)) / math.sqrt(self.dk)
        if attention_mask is not None:
            scores_self = scores_self + ((1 - attention_mask.unsqueeze(1)) * -1e5)
            scores_other = scores_other + ((1 - attention_mask.unsqueeze(1)) * -1e5)
        scores_self = F.softmax(scores_self, dim=-1)
        scores_self = self.drop(scores_self)
        scores_other = F.softmax(scores_other, dim=-1)
        scores_other = self.drop(scores_other)


        # out => (B, L, A)
        out_self = torch.bmm(scores_self, v)
        out_other = torch.bmm(scores_other, v)
        out = out_self + out_other
        return out, scores_self

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.q_self.weight)
        nn.init.xavier_uniform_(self.q_other.weight)
        nn.init.xavier_uniform_(self.v.weight)


class MultiheadAttentionSerial(nn.Module):
    """Serial MultiheadAttention"""

    def __init__(
        self,
        attention_size=512,
        num_heads=8,
        input_size=None,
        dropout=0.1,
        grad_checkpoint=False,
    ):
        super(MultiheadAttentionSerial, self).__init__()
        if input_size is None:
            input_size = attention_size
        self.head_size = int(attention_size / num_heads)
        self.heads = [
            Attention(
                input_size,
                self.head_size,
                dropout=dropout,
                grad_checkpoint=grad_checkpoint,
            )
            for _ in num_heads
        ]

    def forward(self, x, queries=None, values=None, attention_mask=None):
        """
        x : (B, L, D)
        queries : (B, L, D)
        values : (B, L, D)
        """
        # list of (B, L, A / H)
        out = [
            h(x, queries=queries, values=values, attention_mask=attention_mask)
            for h in self.heads
        ]

        # (B, L, A)
        out = torch.cat(out, dim=-1)
        return out


class MultiheadAttentionParallel(nn.Module):
    def __init__(
        self,
        attention_size=512,
        num_heads=8,
        input_size=None,
        dropout=0.1,
        grad_checkpoint=False,
    ):
        super(MultiheadAttentionParallel, self).__init__()
        if input_size is None:
            input_size = attention_size
        self.dk = input_size
        self.num_heads = num_heads
        self.head_size = int(attention_size / num_heads)
        self.attention_size = attention_size
        self.grad_checkpoint = grad_checkpoint
        self.k = nn.Linear(input_size, attention_size, bias=False)
        self.q = nn.Linear(input_size, attention_size, bias=False)
        self.v = nn.Linear(input_size, attention_size, bias=False)
        self.output = FF(
            attention_size,
            attention_size,
            activation="none",
            layer_norm=True,
            dropout=dropout,
        )
        self.drop = nn.Dropout(dropout)
        self._reset_parameters()

    def _split_heads(self, x):
        """
        x => (B, L, A)
        out => (B, H, L, A/H)
        """
        batch_size, max_length, _ = x.size()
        return x.view(batch_size, max_length, self.num_heads, self.head_size).permute(
            0, 2, 1, 3
        )

    def _merge_heads(self, x):
        """
        x => (B, H, L, A/H)
        out => (B, L, A)
        """
        batch_size, _, max_length, _ = x.size()
        # x => (B, L, H, A/H)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, max_length, -1)

    def forward(self, x, queries=None, values=None, attention_mask=None):
        """
        x : (B, L, D)
        queries : (B, L, D)
        values : (B, L, D)
        """
        if queries is None:
            queries = x
        if values is None:
            values = x
        k = self._split_heads(self.k(x))  # (B, H, L, A/H)
        q = self._split_heads(self.q(queries))  # (B, H, L, A/H)
        v = self._split_heads(self.v(values))  # (B, H, L, A/H)

        # scores => (B, H, L, L)
        if self.grad_checkpoint:
            scores = checkpoint(calc_scores(self.dk), q, k)
        else:
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dk)
        if attention_mask is not None:
            scores = scores + ((1 - attention_mask.unsqueeze(1)) * -1e5)
        scores = F.softmax(scores, dim=-1)
        scores = self.drop(scores)

        # out => (B, H, L, A/H)
        out = self._merge_heads(torch.matmul(scores, v))
        out = self.output(out)
        return out

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.v.weight)
        nn.init.xavier_uniform_(self.output.fc.weight)
        nn.init.constant_(self.output.fc.bias, 0.0)


class MultiheadCoAttention(nn.Module):
    def __init__(
        self,
        attention_size=512,
        num_heads=8,
        input_size=None,
        cross_size=None,
        dropout=0.1,
        grad_checkpoint=False,
    ):
        super(MultiheadCoAttention, self).__init__()

        if input_size is None:
            input_size = attention_size

        if cross_size is None:
            cross_size = attention_size

        self.dk = input_size
        self.num_heads = num_heads
        self.head_size = int(attention_size / num_heads)
        self.attention_size = attention_size
        self.grad_checkpoint = grad_checkpoint
        self.k = nn.Linear(input_size, attention_size, bias=False)
        self.q = nn.Linear(cross_size, attention_size, bias=False)
        self.v = nn.Linear(input_size, attention_size, bias=False)
        # self.output_self = FF(
        #     attention_size,
        #     attention_size,
        #     activation="none",
        #     layer_norm=True,
        #     dropout=dropout,
        # )
        # self.output_other = FF(
        #     attention_size,
        #     attention_size,
        #     activation="none",
        #     layer_norm=True,
        #     dropout=dropout,
        # )

        self.out = nn.Linear(attention_size, attention_size)

        self.drop = nn.Dropout(dropout)
        self._reset_parameters()

    def _split_heads(self, x):
        """
        x => (B, L, A)
        out => (B, H, L, A/H)
        """
        batch_size, max_length, _ = x.size()

        return x.view(batch_size, max_length, self.num_heads, self.head_size).permute(
            0, 2, 1, 3
        )

    def _merge_heads(self, x):
        """
        x => (B, H, L, A/H)
        out => (B, L, A)
        """
        batch_size, _, max_length, _ = x.size()
        # x => (B, L, H, A/H)
        x = x.permute(0, 2, 1, 3).contiguous()

        return x.view(batch_size, max_length, -1)

    def forward(self, x, y, values=None, attention_mask=None):
        """
        x : (B, L, D)
        queries : (B, L, D)
        values : (B, L, D)
        """
        if values is None:
            values = y
        k = self._split_heads(self.k(y))  # (B, H, L, A/H)
        q = self._split_heads(self.q(x))  # (B, H, L, A/H)
        v = self._split_heads(self.v(y))  # (B, H, L, A/H)

        # scores => (B, H, L, L)

        if self.grad_checkpoint:
            scores = checkpoint(calc_scores(self.dk), q, k)
        else:
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dk)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)
            scores = scores + ((1 - attention_mask) * -1e5)
        scores = F.softmax(scores, dim=-1)
        scores = self.drop(scores)

        # out => (B, H, L, A/H)
        out = self._merge_heads(torch.matmul(scores, v))

        out = self.out(out) + x

        return out

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.v.weight)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.constant_(self.out.bias, 0.0)



MultiheadAttention = MultiheadAttentionParallel
