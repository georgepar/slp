import torch
from torch.nn.utils.rnn import pad_sequence

from slp.config import SPECIAL_TOKENS
from slp.modules.util import pad_mask, subsequent_mask
from slp.util import mktensor, shift_tensor


class SequenceClassificationCollator(object):
    def __init__(self, pad_indx=0, device='cpu'):
        self.pad_indx = pad_indx
        self.device = device

    def __call__(self, batch):
        inputs, targets = map(list, zip(*batch))
        lengths = torch.tensor([len(s) for s in inputs], device=self.device)
        # Pad and convert to tensor
        inputs = (pad_sequence(inputs,
                               batch_first=True,
                               padding_value=self.pad_indx)
                  .to(self.device))
        targets = mktensor(targets, device=self.device, dtype=torch.long)
        return inputs, targets.to(self.device), lengths


class TransformerLMCollator(object):
    def __init__(self, pad_indx=0, device='cpu'):
        self.pad_indx = pad_indx
        self.device = device

    def __call__(self, batch):
        inputs = batch
        targets = [shift_tensor(x, n=1) for x in inputs]
        lengths = torch.tensor([len(s) for s in inputs], device=self.device)
        max_length = torch.max(lengths)
        pad_m = pad_mask(lengths, max_length=max_length)
        sub_m = subsequent_mask(max_length)
        # Pad and convert to tensor
        inputs = (pad_sequence(inputs,
                               batch_first=True,
                               padding_value=self.pad_indx)
                  .to(self.device))
        targets = (pad_sequence(inputs,
                                batch_first=True,
                                padding_value=self.pad_indx)
                   .to(self.device))
        return inputs, targets, pad_m, sub_m
