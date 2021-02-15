import torch
from torch.nn.utils.rnn import pack_padded_sequence , pad_sequence

from slp.modules.util import pad_mask, subsequent_mask
from slp.util import mktensor
from slp.data.therapy_title import pad_sequence as pad_sequence1

class SequenceClassificationCollator(object):
    def __init__(self, pad_indx=0, device='cpu'):
        self.pad_indx = pad_indx
        self.device = device

    def __call__(self, batch):
#        import pdb; pdb.set_trace()
        padding_len=150
        inputs, titles, targets = map(list, zip(*batch))
        number_of_sentences = torch.tensor([len(s) for s in inputs], device=self.device)
        length_of_sentences = ([torch.tensor([len(s) if len(s)<padding_len else padding_len for s in inp]) for inp in inputs])

        inputs = [pad_sequence1(i, padding_len=150, batch_first=True, padding_value=0) for i in inputs]


        # Pad and convert to tensor
        inputs = (pad_sequence(inputs,
                               batch_first=True,
                               padding_value=self.pad_indx)
                  .to(self.device))
        length_of_sentences = pad_sequence1(length_of_sentences, padding_len=inputs.shape[1], batch_first=True, padding_value=1)
       
        lengths_title = torch.tensor([len(t) for t in titles], device=self.device)
        titles = (pad_sequence(titles,
                               batch_first=True,
                               padding_value=self.pad_indx)
                  .to(self.device))

        targets = mktensor(targets, device=self.device, dtype=torch.long)
        return inputs, titles, targets.to(self.device), number_of_sentences, length_of_sentences, lengths_title


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
