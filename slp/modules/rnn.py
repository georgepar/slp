import torch.nn as nn

from slp.modules.unpack import PadPackedSequence


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True,
                 layers=1, bidirectional=False, dropout=0,
                 rnn_type='lstm', unpack=True):
        super(RNN, self).__init__()
        rnn_cls = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.rnn = rnn_cls(input_size,
                           hidden_size,
                           batch_first=False,
                           num_layers=layers,
                           bidirectional=bidirectional)
        self.drop = nn.Dropout(dropout)
        self.unpack = None
        if unpack:
            self.unpack = PadPackedSequence(batch_first=batch_first)

    def forward(self, x, lengths):
        self.rnn.flatten_parameters()
        out, _ = self.rnn(x)
        if self.unpack is not None:
            out = self.unpack(out, lengths)
        out = self.drop(out)
        last_hidden = out[:, -1, :]
        return out, last_hidden
