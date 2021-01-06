import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from slp.modules.util import pad_mask, subsequent_mask
from slp.util import mktensor


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


class TransformerCollator(object):
    def __init__(self, pad_indx=0, device="cpu"):
        self.pad_indx = pad_indx
        self.device = device

    def pad_and_mask(self, tensors):
        lengths = torch.tensor([len(s) for s in tensors], device=self.device)
        max_length = torch.max(lengths)
        pad_m = pad_mask(lengths, max_length=max_length, device=self.device)
        sub_m = subsequent_mask(max_length)
        tensors = pad_sequence(
            tensors, batch_first=True, padding_value=self.pad_indx
        ).to(self.device)

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
    def __init__(self, pad_indx=0, device="cpu", batch_first=True):
        self.seq_collator = SequenceClassificationCollator(
            pad_indx=pad_indx, device=device
        )
        self.batch_first = batch_first
        self.device = device

    def __call__(self, batch):
        inputs, targets, lengths = self.seq_collator(batch)
        inputs = pack_padded_sequence(
            inputs, lengths, batch_first=self.batch_first, enforce_sorted=False
        )

        return inputs, targets.to(self.device), lengths[inputs.sorted_indices]


class MOSICollator(object):
    def __init__(
        self,
        modalities=("text", "audio"),
        binary=True,
        pad_indx=0,
        device="cpu",
        max_length=-1,
    ):
        self.pad_indx = pad_indx
        self.device = device
        self.modalities = list(modalities)
        self.binary = binary
        self.max_length = max_length
        self.target_dtype = torch.long if binary else torch.float

    def extract_label(self, l):
        return int(l[0]) if self.binary else l[0]

    def extract_sequence(self, s):
        return s[self.max_length :] if self.max_length > 0 else s

    def __call__(self, batch):
        data = {}
        #data["lengths"] = torch.tensor(
        #    [len(self.extract_sequence(b[self.modalities[0]])) for b in batch], device=self.device
        #)

        for m in self.modalities:
            inputs = [self.extract_sequence(b[m]) for b in batch]
            data[m] = pad_sequence(
                inputs, batch_first=True, padding_value=self.pad_indx
            ).to(self.device)

        data["lengths"] = torch.tensor(
            [len(s) for s in data[self.modalities[0]]], device=self.device
        )

        targets = [self.extract_label(b["label"]) for b in batch]
        targets = mktensor(targets, device=self.device, dtype=self.target_dtype)

        return data, targets.to(self.device)
