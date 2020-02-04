import torch.nn as nn
import torch

from slp.modules.attention import Attention
from slp.modules.embed import Embed
from slp.modules.helpers import PackSequence, PadPackedSequence

from slp.modules.util import pad_mask


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True,
                 layers=1, bidirectional=False, merge_bi='cat', dropout=0,
                 rnn_type='lstm', packed_sequence=True, device='cpu'):

        super(RNN, self).__init__()
        self.device = device
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.merge_bi = merge_bi
        self.rnn_type = rnn_type.lower()

        rnn_cls = nn.LSTM if self.rnn_type == 'lstm' else nn.GRU
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

    def _merge_bi(self, forward, backward):
        if self.merge_bi == 'sum':
            return forward + backward
        return torch.cat((forward, backward), dim=-1)

    def _select_last_unpadded(self, out, lengths):
        gather_dim = 1 if self.batch_first else 0
        gather_idx = ((lengths - 1)  # -1 to convert to indices
                      .unsqueeze(1)  # (B) -> (B, 1)
                      .expand((-1, self.hidden_size))  # (B, 1) -> (B, H)
                      # (B, 1, H) if batch_first else (1, B, H)
                      .unsqueeze(gather_dim))
        # Last forward for real length or seq (unpadded tokens)
        last_out = out.gather(gather_dim, gather_idx).squeeze(gather_dim)
        return last_out

    def _final_output(self, out, lengths):
        # Collect last hidden state
        # Code adapted from https://stackoverflow.com/a/50950188
        if not self.bidirectional:
            return self._select_last_unpadded(out, lengths)

        forward, backward = (out[..., :self.hidden_size],
                             out[..., self.hidden_size:])
        # Last backward corresponds to first token
        last_backward_out = (backward[:, 0, :]
                             if self.batch_first
                             else backward[0, ...])
        # Last forward for real length or seq (unpadded tokens)
        last_forward_out = self._select_last_unpadded(forward, lengths)
        return self._merge_bi(last_forward_out, last_backward_out)

    def forward(self, x, lengths):
        self.rnn.flatten_parameters()
        if self.packed_sequence:
            x, lengths = self.pack(x, lengths)
        out, hidden = self.rnn(x)
        if self.packed_sequence:
            out = self.unpack(out, lengths)
        out = self.drop(out)

        last_timestep = self._final_output(out, lengths)
        return out, last_timestep, hidden


class WordRNN(nn.Module):
    def __init__(
            self, hidden_size, embeddings,
            embeddings_dropout=.1, finetune_embeddings=False,
            batch_first=True, layers=1, bidirectional=False, merge_bi='cat',
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
            batch_first=batch_first, layers=layers, merge_bi=merge_bi,
            bidirectional=bidirectional, dropout=dropout,
            rnn_type=rnn_type, packed_sequence=packed_sequence)
        self.out_size = hidden_size if not bidirectional else 2 * hidden_size
        self.attention = None
        if attention:
            self.attention = Attention(
                attention_size=self.out_size, dropout=dropout)

    def forward(self, x, lengths):
        x = self.embed(x)
        out, last_hidden, _ = self.rnn(x, lengths)
        if self.attention is not None:
            out, _ = self.attention(
                out, attention_mask=pad_mask(lengths, device=self.device))
            out = out.sum(1)
        else:
            out = last_hidden
        return out


class WordRNN2(nn.Module):
    def __init__(
            self, embedding_dim, vocab_size, hidden_size, embeddings=None,
            embeddings_dropout=.1, finetune_embeddings=False,
            batch_first=True, layers=1, bidirectional=False, merge_bi='cat',
            dropout=0.1, rnn_type='lstm', packed_sequence=True,
            attention=False, device='cpu'):
        super(WordRNN2, self).__init__()
        self.device = device
        self.embed = Embed(embedding_dim=embedding_dim,
                           num_embeddings=vocab_size,
                           embeddings=embeddings,
                           dropout=embeddings_dropout,
                           trainable=finetune_embeddings)
        self.rnn = RNN(
            input_size=embedding_dim, hidden_size=hidden_size,
            batch_first=batch_first, layers=layers, merge_bi=merge_bi,
            bidirectional=bidirectional, dropout=dropout,
            rnn_type=rnn_type, packed_sequence=packed_sequence)
        self.out_size = hidden_size if not bidirectional else 2 * hidden_size
        self.attention = None
        if attention:
            self.attention = Attention(
                attention_size=self.out_size, dropout=dropout)

    def forward(self, x, lengths):
        x = self.embed(x)
        out, last_hidden, _ = self.rnn(x, lengths)
        if self.attention is not None:
            out, _ = self.attention(
                out, attention_mask=pad_mask(lengths, device=self.device))
            out = out.sum(1)
        else:
            out = last_hidden
        return out