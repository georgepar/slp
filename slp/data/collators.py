import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from slp.util.pytorch import mktensor, pad_mask, subsequent_mask


class SequenceClassificationCollator(object):
    def __init__(self, pad_indx=0, device="cpu"):
        self.pad_indx = pad_indx
        self.device = device

    def __call__(self, batch):
        inputs, targets = map(list, zip(*batch))
        lengths = torch.tensor([len(s) for s in inputs], device=self.device)
        # Pad and convert to tensor
        inputs = pad_sequence(inputs, batch_first=True, padding_value=self.pad_indx).to(
            self.device
        )
        targets = mktensor(targets, device=self.device, dtype=torch.long)
        return inputs, targets.to(self.device), lengths


class Seq2SeqCollator(object):
    def __init__(self, pad_indx=0, device="cpu"):
        self.pad_indx = pad_indx
        self.device = device

    @staticmethod
    def get_inputs_and_targets(batch):
        return inputs, targets

    def __call__(self, batch):
        inputs, targets = map(list, zip(*batch))
        lengths_inputs = torch.tensor([len(s) for s in inputs], device=self.device)
        lengths_targets = torch.tensor([len(s) for s in targets], device=self.device)

        inputs = pad_sequence(inputs, batch_first=True, padding_value=self.pad_indx).to(
            self.device
        )

        targets = pad_sequence(
            targets, batch_first=True, padding_value=self.pad_indx
        ).to(self.device)

        return inputs, targets, lengths_inputs, lengths_targets