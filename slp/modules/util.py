import copy
import torch


def repeat_layer(l, times):
    return [l] + [copy.deepcopy(l) for _ in range(times - 1)]


def pad_mask(lengths, max_length=None):
    """lengths is a torch tensor
    """
    if max_length is None:
        max_length = torch.max(lengths)
    idx = torch.arange(0, max_length).unsqueeze(0)
    mask = (idx < lengths.unsqueeze(1)).float()
    return mask


def subsequent_mask(max_length):
    mask = torch.ones(max_length, max_length)
    return mask.triu().t().unsqueeze(0).contiguous()
