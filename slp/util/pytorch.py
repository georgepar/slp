import copy
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from loguru import logger
from typing import Optional

from slp.util import system
from slp.util import types

from typing import cast, Callable, Optional, Tuple


class PadPackedSequence(nn.Module):
    """Some Information about PadPackedSequence"""

    def __init__(self, batch_first=True):
        super(PadPackedSequence, self).__init__()
        self.batch_first = batch_first

    def forward(self, x, lengths):
        max_length = lengths.max().item()
        x, _ = pad_packed_sequence(
            x, batch_first=self.batch_first, total_length=max_length
        )
        return x


class PackSequence(nn.Module):
    def __init__(self, batch_first=True):
        super(PackSequence, self).__init__()
        self.batch_first = batch_first

    def forward(self, x, lengths):
        x = pack_padded_sequence(
            x, lengths, batch_first=self.batch_first, enforce_sorted=False
        )
        lengths = lengths[x.sorted_indices]
        return x, lengths


def repeat_layer(l: torch.nn.Module, times: int):
    return [l] + [copy.deepcopy(l) for _ in range(times - 1)]


def pad_mask(lengths: torch.Tensor, max_length: Optional[int] = None):
    """lengths is a torch tensor"""
    if max_length is None:
        max_length = cast(int, torch.max(lengths).item())
    max_length = cast(int, max_length)
    idx = torch.arange(0, max_length, device=lengths.device).unsqueeze(0)
    mask = (idx < lengths.unsqueeze(1)).float()
    return mask


def subsequent_mask(max_length: int):
    mask = torch.ones(max_length, max_length)
    # Ignore typecheck because pytorch types are incomplete
    return mask.triu().t().unsqueeze(0).contiguous()  # type: ignore


def sort_sequences(
    inputs: torch.Tensor, lengths: torch.Tensor
) -> (Tuple[torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]):
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


def to_device(
    tt: torch.Tensor, device: Optional[types.Device] = "cpu", non_blocking: bool = False
) -> torch.Tensor:
    return tt.to(device, non_blocking=non_blocking)


def t_(
    data: types.NdTensor,
    dtype: torch.dtype = torch.float,
    device: Optional[types.Device] = "cpu",
    requires_grad: bool = False,
) -> torch.Tensor:
    """Convert a list or numpy array to torch tensor. If a torch tensor
    is passed it is cast to  dtype, device and the requires_grad flag is
    set IN PLACE.

    Args:
        data: (list, np.ndarray, torch.Tensor): Data to be converted to
            torch tensor.
        dtype: (torch.dtype): The type of the tensor elements
            (Default value = torch.float)
        device: (torch.device, str): Device where the tensor should be
            (Default value = 'cpu')
        requires_grad: bool): Trainable tensor or not? (Default value = False)

    Returns:
        (torch.Tensor): A tensor of appropriate dtype, device and
            requires_grad containing data

    """
    if isinstance(device, str):
        device = torch.device(device)

    tt = torch.as_tensor(data, dtype=dtype, device=device).requires_grad_(requires_grad)
    return tt


def t(
    data: types.NdTensor,
    dtype: torch.dtype = torch.float,
    device: types.Device = "cpu",
    requires_grad: bool = False,
) -> torch.Tensor:
    """Convert a list or numpy array to torch tensor. If a torch tensor
    is passed it is cast to  dtype, device and the requires_grad flag is
    set. This always copies data.

    Args:
        data: (list, np.ndarray, torch.Tensor): Data to be converted to
            torch tensor.
        dtype: (torch.dtype): The type of the tensor elements
            (Default value = torch.float)
        device: (torch.device, str): Device where the tensor should be
            (Default value = 'cpu')
        requires_grad: (bool): Trainable tensor or not? (Default value = False)

    Returns:
        (torch.Tensor): A tensor of appropriate dtype, device and
            requires_grad containing data

    """
    tt = torch.tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)
    return tt


def mktensor(
    data: types.NdTensor,
    dtype: torch.dtype = torch.float,
    device: types.Device = "cpu",
    requires_grad: bool = False,
    copy: bool = True,
) -> torch.Tensor:
    """Convert a list or numpy array to torch tensor. If a torch tensor
        is passed it is cast to  dtype, device and the requires_grad flag is
        set. This can copy data or make the operation in place.

    Args:
        data: (list, np.ndarray, torch.Tensor): Data to be converted to
            torch tensor.
        dtype: (torch.dtype): The type of the tensor elements
            (Default value = torch.float)
        device: (torch.device, str): Device where the tensor should be
            (Default value = 'cpu')
        requires_grad: (bool): Trainable tensor or not? (Default value = False)
        copy: (bool): If false creates the tensor inplace else makes a copy
            (Default value = True)

    Returns:
        (torch.Tensor): A tensor of appropriate dtype, device and
            requires_grad containing data

    """
    tensor_factory = t if copy else t_
    return tensor_factory(data, dtype=dtype, device=device, requires_grad=requires_grad)


def from_checkpoint(
    checkpoint_file: Optional[str],
    obj: types.ModuleOrOptimizer,
    map_location: Optional[types.Device] = None,
) -> types.ModuleOrOptimizer:  # noqa: E501
    if checkpoint_file is None:
        return obj

    if not system.is_file(checkpoint_file):
        logger.warning(
            f"The checkpoint {checkpoint_file} you are trying to load "
            "does not exist. Continuing without loading..."
        )
        return obj

    state_dict = torch.load(checkpoint_file, map_location=map_location)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    obj.load_state_dict(state_dict)
    return obj


def rotate_tensor(l: torch.Tensor, n: int = 1) -> torch.Tensor:
    return torch.cat((l[n:], l[:n]))


def shift_tensor(l: torch.Tensor, n: int = 1) -> torch.Tensor:
    out = rotate_tensor(l, n=n)
    out[-n:] = 0
    return out
