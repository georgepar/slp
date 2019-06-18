import torch.nn as nn
import torch
from slp.modules.attention import Attention
from slp.modules.embed import Embed
from slp.modules.helpers import PackSequence, PadPackedSequence
from slp.modules.util import pad_mask


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True,
                 layers=1, bidirectional=False, dropout=0,
                 rnn_type='lstm', packed_sequence=True, device='cpu'):
        super(RNN, self).__init__()
        rnn_cls = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.rnn = rnn_cls(input_size,
                           hidden_size,
                           batch_first=batch_first,
                           num_layers=layers,
                           bidirectional=bidirectional)
        self.drop = nn.Dropout(dropout)
        self.packed_sequence = packed_sequence
        if packed_sequence:
            self.pack = PackSequence(batch_first=batch_first)
            self.unpack = PadPackedSequence(batch_first=batch_first)
        self.device = device
        self.bidirectional = bidirectional

    def forward(self, x, lengths):
        self.rnn.flatten_parameters()
        if self.packed_sequence:
            x, lengths = self.pack(x, lengths)
        out, (hidden, cell) = self.rnn(x)
        if self.packed_sequence:
            out = self.unpack(out, lengths)
        out = self.drop(out)
        last_hidden = (
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            if self.bidirectional
            else hidden[-1, :, :])
        return out, last_hidden, hidden


class WordRNN(nn.Module):
    def __init__(
            self, hidden_size, embeddings,
            embeddings_dropout=.1, finetune_embeddings=False,
            batch_first=True, layers=1, bidirectional=False,
            dropout=0.1, rnn_type='lstm', packed_sequence=True,
            attention=False, device='cpu'):
        super(WordRNN, self).__init__()
        self.device = device
        self.embed = Embed(embeddings.shape[0],
                           embeddings.shape[1],
                           embeddings=embeddings,
                           dropout=embeddings_dropout,
                           trainable=finetune_embeddings)
        self.rnn = RNN(
            embeddings.shape[1], hidden_size,
            batch_first=batch_first, layers=layers,
            bidirectional=bidirectional, dropout=dropout,
            rnn_type=rnn_type, packed_sequence=packed_sequence)
        self.out_size = hidden_size if not bidirectional else 2 * hidden_size
        self.attention = None
        if attention:
            self.attention = Attention(
                attention_size=self.out_size, dropout=dropout)

    def forward(self, x, lengths):
        x = self.embed(x)
        out, last_hidden, hidden = self.rnn(x, lengths)
        if self.attention is not None:
            out, _ = self.attention(
                out, attention_mask=pad_mask(lengths, device=self.device))
            out = out.sum(1)
        else:
            out = last_hidden
        return out
