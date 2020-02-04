import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from slp.modules.util import pad_mask, subsequent_mask
from slp.util import mktensor


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

class HRED_Collator(object):
    def __init__(self, pad_indx=0, device='cpu'):
        self.pad_indx = pad_indx
        self.device = device

    def __call__(self, batch):
        inputs1, inputs2, inputs3 = map(list, zip(*batch))
        lengths1 = torch.tensor([len(s) for s in inputs1], device=self.device)
        lengths2 = torch.tensor([len(s) for s in inputs2], device=self.device)
        lengths3 = torch.tensor([len(s) for s in inputs3], device=self.device)
        # Pad and convert to tensor
        inputs1 = (pad_sequence(inputs1,
                               batch_first=True,
                               padding_value=self.pad_indx)
                  .to(self.device))
        inputs2 = (pad_sequence(inputs2,
                               batch_first=True,
                               padding_value=self.pad_indx)
                  .to(self.device))
        inputs3 = (pad_sequence(inputs3,
                               batch_first=True,
                               padding_value=self.pad_indx)
                  .to(self.device))
        return inputs1, lengths1, inputs2, lengths2, inputs3, lengths3

class TransformerCollator(object):
    def __init__(self, pad_indx=0, device='cpu'):
        self.pad_indx = pad_indx
        self.device = device

    def pad_and_mask(self, tensors):
        lengths = torch.tensor([len(s) for s in tensors],
                               device=self.device)
        max_length = torch.max(lengths)
        pad_m = pad_mask(lengths, max_length=max_length, device=self.device)
        sub_m = subsequent_mask(max_length)
        tensors = (pad_sequence(tensors,
                                batch_first=True,
                                padding_value=self.pad_indx)
                   .to(self.device))
        return tensors, pad_m, sub_m

    @staticmethod
    def get_inputs_and_targets(batch):
        inputs, targets = map(list, zip(*batch))
        return inputs, targets

    def __call__(self, batch):
        inputs, targets = self.get_inputs_and_targets(batch)
        inputs, pad_m_inputs, _ = self.pad_and_mask(inputs)
        targets, pad_m_targets, sub_m = self.pad_and_mask(targets)
        mask_targets = pad_m_targets.unsqueeze(-2) * sub_m
        mask_inputs = pad_m_inputs.unsqueeze(-2)
        return inputs, targets, mask_inputs, mask_targets


class PackedSequenceCollator(object):
    def __init__(self, pad_indx=0, device='cpu', batch_first=True):
        self.seq_collator = SequenceClassificationCollator(
            pad_indx=pad_indx, device=device)
        self.batch_first = batch_first
        self.device = device

    def __call__(self, batch):
        inputs, targets, lengths = self.seq_collator(batch)
        inputs = pack_padded_sequence(
            inputs, lengths,
            batch_first=self.batch_first,
            enforce_sorted=False)
        return inputs, targets.to(self.device), lengths[inputs.sorted_indices]
