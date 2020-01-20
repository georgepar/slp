import copy
import torch
import torch.nn as nn

from typing import cast, Callable, Optional, Tuple


def repeat_layer(l: torch.nn.Module, times: int):
    return [l] + [copy.deepcopy(l) for _ in range(times - 1)]


def pad_mask(lengths: torch.Tensor,
             max_length: Optional[int] = None,
             device='cpu'):
    """lengths is a torch tensor
    """
    if max_length is None:
        max_length = cast(int, torch.max(lengths).item())
    max_length = cast(int, max_length)
    idx = torch.arange(0, max_length).unsqueeze(0).to(device)
    mask = (idx < lengths.unsqueeze(1)).float()
    return mask


def subsequent_mask(max_length: int):
    mask = torch.ones(max_length, max_length)
    # Ignore typecheck because pytorch types are incomplete
    return mask.triu().t().unsqueeze(0).contiguous()  # type: ignore


def sort_sequences(inputs: torch.Tensor, lengths: torch.Tensor) -> (
        Tuple[torch.Tensor, torch.Tensor,
              Callable[[torch.Tensor], torch.Tensor]]):
    """Sort sequences according to lengths (descending)
    Args:
        inputs (torch.Tensor): input sequences, size [B, T, D]
        lengths (torch.Tensor): length of each sequence, size [B]
    """
    lengths_sorted, sorted_idx = lengths.sort(descending=True)
    _, unsorted_idx = sorted_idx.sort()

    def unsort(t: torch.Tensor) -> torch.Tensor:
        return t[unsorted_idx]

    return inputs[sorted_idx], lengths_sorted, unsort



class Maxout(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self._pool_size = pool_size

    def forward(self, x):
        assert x.shape[-1] % self._pool_size == 0, \
            'Wrong input last dim size ({}) for Maxout({})'.format(x.shape[-1], self._pool_size)
        m, i = x.view(*x.shape[:-1], x.shape[-1] // self._pool_size, self._pool_size).max(-1)
        return m

def max_out(x):
    # make sure s2 is even and that the input is 2 dimension
    if len(x.size()) == 2:
        s1, s2 = x.size()
        x = x.unsqueeze(1)
        x = x.view(s1, s2 // 2, 2)
        x, _ = torch.max(x, 2)

    elif len(x.size()) == 3:
        s1, s2, s3 = x.size()
        x = x.unsqueeze(1)
        x = x.view(s1, s2, s3 // 2, 2)
        x, _ = torch.max(x, 3)

    return x

class Maxout2(nn.Module):

    def __init__(self, d_in, d_out, pool_size):
        super().__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = nn.Linear(d_in, d_out * pool_size)


    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(max_dim)
        return m