import copy
import torch

import torch

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from loguru import logger
from typing import Optional

from slp.util import system
from slp.util import types

from typing import cast, Callable, Optional, Tuple, Union, List


class PadPackedSequence(nn.Module):
    def __init__(self, batch_first: bool = True):
        """Wrap sequence padding in nn.Module

        Args:
            batch_first (bool, optional): Use batch first representation. Defaults to True.
        """
        super(PadPackedSequence, self).__init__()
        self.batch_first = batch_first

    def forward(
        self, x: torch.nn.utils.rnn.PackedSequence, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Convert packed sequence to padded sequence

        Args:
            x (torch.nn.utils.rnn.PackedSequence): Packed sequence
            lengths (torch.Tensor): Sorted original sequence lengths

        Returns:
            torch.Tensor: Padded sequence
        """
        max_length = lengths.max().item()
        out, _ = pad_packed_sequence(
            x, batch_first=self.batch_first, total_length=max_length  # type: ignore
        )
        return out  # type: ignore


class PackSequence(nn.Module):
    def __init__(self, batch_first: bool = True):
        """Wrap sequence packing in nn.Module

        Args:
            batch_first (bool, optional): Use batch first representation. Defaults to True.
        """
        super(PackSequence, self).__init__()
        self.batch_first = batch_first

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.nn.utils.rnn.PackedSequence, torch.Tensor]:
        """Pack a padded sequence and sort lengths

        Args:
            x (torch.Tensor): Padded tensor
            lengths (torch.Tensor): Original lengths befor padding

        Returns:
            Tuple[torch.nn.utils.rnn.PackedSequence, torch.Tensor]: (packed sequence, sorted lengths)
        """
        out: torch.nn.utils.rnn.PackedSequence = pack_padded_sequence(
            x, lengths, batch_first=self.batch_first, enforce_sorted=False
        )
        lengths = lengths[out.sorted_indices]
        return out, lengths


def repeat_layer(l: nn.Module, times: int) -> List[nn.Module]:
    """Clone a layer multiple times

    Args:
        l (nn.Module): nn.Module to stack
        times (int): Times to clone

    Returns:
        List[nn.Module]: List of identical clones of input layer
    """
    return [l] + [copy.deepcopy(l) for _ in range(times - 1)]


def pad_mask(
    lengths: torch.Tensor, max_length: Optional[Union[torch.Tensor, int]] = None
) -> torch.Tensor:
    """Generate mask for padded tokens

    Args:
        lengths (torch.Tensor): Original sequence lengths before padding
        max_length (Optional[Union[torch.Tensor, int]], optional): Maximum sequence length. Defaults to None.

    Returns:
        torch.Tensor: padding mask
    """
    if max_length is None:
        max_length = cast(int, torch.max(lengths).item())
    max_length = cast(int, max_length)
    idx = torch.arange(0, max_length, device=lengths.device).unsqueeze(0)
    mask: torch.Tensor = (idx < lengths.unsqueeze(1)).float()
    return mask


def subsequent_mask(max_length: int) -> torch.Tensor:
    """Generate subsequent (lower triangular) mask for transformer autoregressive tasks

    Args:
        max_length (int): Maximum sequence length

    Returns:
        torch.Tensor: The subsequent mask
    """
    mask = torch.ones(max_length, max_length)
    # Ignore typecheck because pytorch types are incomplete
    return mask.triu().t().unsqueeze(0).contiguous()  # type: ignore


def sort_sequences(
    inputs: torch.Tensor, lengths: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    """Sort sequences according to lengths (descending)

    Args:
        inputs (torch.Tensor): input sequences, size [B, T, D]
        lengths (torch.Tensor): length of each sequence, size [B]

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.tensor]]:
            (sorted inputs, sorted lengths, function to revert inputs and lengths to unsorted state)
    """
    lengths_sorted, sorted_idx = lengths.sort(descending=True)
    _, unsorted_idx = sorted_idx.sort()

    def unsort(t: torch.Tensor) -> torch.Tensor:
        """Restore original unsorted sequence"""
        return t[unsorted_idx]

    return inputs[sorted_idx], lengths_sorted, unsort


def to_device(
    tt: torch.Tensor, device: Optional[types.Device] = "cpu", non_blocking: bool = False
) -> torch.Tensor:
    """Send a tensor to a device

    Args:
        tt (torch.Tensor): input tensor
        device (Optional[types.Device], optional): Output device. Defaults to "cpu".
        non_blocking (bool, optional): Use blocking or non-blocking memory transfer. Defaults to False.

    Returns:
        torch.Tensor: Tensor in the desired device
    """
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
    map_location: Optional[types.Device] = "cpu",
    dataparallel: bool = False,
) -> types.ModuleOrOptimizer:
    """Load model or optimizer from saved state_dict

    Args:
        checkpoint_file (Optional[str]): File containing the state dict
        obj (types.ModuleOrOptimizer): Module or optimizer instance to load the checkpoint
        map_location (Optional[types.Device], optional): Where to load. Defaults to "cpu".
        dataparallel (bool, optional): If data parallel remove leading "module." from statedict keys. Defaults to False.

    Returns:
        types.ModuleOrOptimizer: Loaded module or optimizer
    """
    if checkpoint_file is None:
        return obj

    if not system.is_file(checkpoint_file):
        logger.warning(
            f"The checkpoint {checkpoint_file} you are trying to load "
            "does not exist. Continuing without loading..."
        )
        return obj

    state_dict = torch.load(checkpoint_file, map_location=map_location)
    if dataparallel:
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    obj.load_state_dict(state_dict)
    return obj


def rotate_tensor(l: torch.Tensor, n: int = 1) -> torch.Tensor:
    """Roate tensor by n positions to the right

    Args:
        l (torch.Tensor): input tensor
        n (int, optional): positions to rotate. Defaults to 1.

    Returns:
        torch.Tensor: rotated tensor
    """
    return torch.cat((l[n:], l[:n]))


def shift_tensor(l: torch.Tensor, n: int = 1) -> torch.Tensor:
    """Shift tensor by n positions

    Args:
        l (torch.Tensor): input tensor
        n (int, optional): positions to shift. Defaults to 1.

    Returns:
        torch.Tensor: shifted tensor
    """
    out = rotate_tensor(l, n=n)
    out[-n:] = 0
    return out
