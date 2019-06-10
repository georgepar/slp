import torch
from torch.nn.utils.rnn import pad_sequence

from slp.util import mktensor
from slp.config import SPECIAL_TOKENS


class SequenceClassificationCollator(object):
    def __init__(self, pad_indx=SPECIAL_TOKENS.PAD.value, device='cpu'):
        self.pad_indx = pad_indx
        self.device = device

    def __call__(self, batch):
        inputs, targets = map(list, zip(*batch))
        lengths = [len(s) for s in inputs]
        lengths = torch.tensor(lengths, device=self.device)
        # Pad and convert to tensor
        inputs = pad_sequence(inputs,
                              batch_first=True,
                              padding_value=self.pad_indx)
        inputs = inputs.to(self.device)
        targets = mktensor(targets, device=self.device, dtype=torch.long)
        return inputs, targets.to(self.device), lengths
