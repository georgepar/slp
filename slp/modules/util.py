import copy
import torch

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
