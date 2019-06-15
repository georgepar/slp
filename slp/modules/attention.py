import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Some Information about Attention"""
    def __init__(self, attention_size=512, input_size=None, dropout=.1):
        super(Attention, self).__init__()
        if input_size is None:
            input_size = attention_size
        self.dk = input_size
        self.k = nn.Linear(input_size, attention_size, bias=False)
        self.q = nn.Linear(input_size, attention_size, bias=False)
        self.v = nn.Linear(input_size, attention_size, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, queries=None, values=None, attention_mask=None):
        '''
        x : (B, L, D)
        queries : (B, L, D)
        values : (B, L, D)
        '''
        if queries is None:
            queries = x
        if values is None:
            values = x
        k = self.k(x)  # (B, L, A)
        q = self.q(queries)  # (B, L, A)
        v = self.v(values)  # (B, L, A)

        # weights => (B, L, L)
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.dk)
        if attention_mask is not None:
            scores = scores * attention_mask
        scores = F.softmax(scores, dim=-1)
        scores = self.drop(scores)

        # out => (B, L, A)
        out = torch.bmm(scores, v)
        return out, scores


class MultiheadAttentionSerial(nn.Module):
    """Serial MultiheadAttention"""
    def __init__(self,
                 attention_size=512,
                 num_heads=8,
                 input_size=None,
                 dropout=.1):
        super(MultiheadAttentionSerial, self).__init__()
        if input_size is None:
            input_size = attention_size
        self.head_size = int(attention_size / num_heads)
        self.heads = [Attention(input_size, self.head_size, dropout=dropout)
                      for _ in num_heads]

    def forward(self, x, queries=None, values=None, attention_mask=None):
        """
        x : (B, L, D)
        queries : (B, L, D)
        values : (B, L, D)
        """
        # list of (B, L, A / H)
        out = [h(x,
                 queries=queries,
                 values=values,
                 attention_mask=attention_mask)
               for h in self.heads]

        # (B, L, A)
        out = torch.cat(out, dim=-1)
        return out


class MultiheadAttentionParallel(nn.Module):
    def __init__(self,
                 attention_size=512,
                 num_heads=8,
                 input_size=None,
                 dropout=.1):
        super(MultiheadAttentionParallel, self).__init__()
        if input_size is None:
            input_size = attention_size
        self.dk = input_size
        self.num_heads = num_heads
        self.head_size = int(attention_size / num_heads)
        self.attention_size = attention_size
        self.k = nn.Linear(input_size, attention_size, bias=False)
        self.q = nn.Linear(input_size, attention_size, bias=False)
        self.v = nn.Linear(input_size, attention_size, bias=False)
        self.drop = nn.Dropout(dropout)

    def _split_heads(self, x):
        """
        x => (B, L, A)
        out => (B, H, L, A/H)
        """
        batch_size, max_length, _ = x.size()
        return (x
                .view(batch_size, max_length,
                      self.num_heads, self.head_size)
                .permute(0, 2, 1, 3))

    def _merge_heads(self, x):
        """
        x => (B, H, L, A/H)
        out => (B, L, A)
        """
        batch_size, max_length, _ = x.size()
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
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dk)
        if attention_mask is not None:
            scores = scores * attention_mask
        scores = F.softmax(scores, dim=-1)
        scores = self.drop(scores)

        # out => (B, H, L, A/H)
        out = self._merge_heads(torch.matmul(scores, v))
        return out


MultiheadAttention = MultiheadAttentionParallel
