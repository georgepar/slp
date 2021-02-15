import torch.nn as nn
from torch.nn.utils.rnn import (
    pack_padded_sequence, pad_packed_sequence)


class PadPackedSequence(nn.Module):
    """Some Information about PadPackedSequence"""
    def __init__(self, batch_first=True):
        super(PadPackedSequence, self).__init__()
        self.batch_first = batch_first

    def forward(self, x, lengths):
#        import pdb; pdb.set_trace()
        max_length = lengths.max().item()
        x, _ = pad_packed_sequence(
            x, batch_first=self.batch_first, total_length=max_length)
        return x


class PackSequence(nn.Module):
    def __init__(self, batch_first=True):
        super(PackSequence, self).__init__()
        self.batch_first = batch_first

    def forward(self, x, lengths):
#        import pdb; pdb.set_trace()
        x = pack_padded_sequence(
            x, lengths,
            batch_first=self.batch_first,
            enforce_sorted=False)
        lengths = lengths[x.sorted_indices]
        return x, lengths
