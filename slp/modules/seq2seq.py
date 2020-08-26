import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from slp.modules.rnn import RNN, StackedGRUCell, StackedLSTMCell
from slp.modules.attention import GlobalAttention

from slp.modules.util import pad_mask


class RNNEncoder(nn.Module):
    def __init__(
        self, hidden_size, vocab_size, embedding_size=256,
        layers=1, dropout=0.1, rnn_type='gru', packed_sequence=True
    ):
        super(RNNEncoder, self).__init__()
        bidirectional = False  # Use unidirectional for now
        batch_first = True
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.encoded_size = hidden_size if not bidirectional else 2 * hidden_size
        self.rnn = RNN(
            embedding_size, hidden_size, batch_first=batch_first,
            layers=layers, bidirectional=bidirectional,
            dropout=dropout, rnn_type=rnn_type,
            packed_sequence=packed_sequence
        )
    
    def forward(self, src, lengths):
        inputs = self.embed(src)
        out, _, hidden = self.rnn(inputs, lengths)
        return out, hidden

"""
class RNNDecoderCell(nn.Module):
    def __init__(
        self, hidden_size, vocab_size, embedding_size=256,
        layers=1, dropout=0.1, attention=False,
        rnn_type='gru', tie=False
    ):
        super(RNNDecoderCell, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.rnn_type = rnn_type
        rnn_cls = StackedLSTMCell if self.rnn_type == 'lstm' else StackedGRUCell
        self.rnn = rnn_cls(embedding_size, hidden_size, num_layers=layers)
        self.attend = attention
        if self.attend:
            self.attention = GlobalAttention(hidden_size, hidden_size)
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_size, vocab_size)

        if tie:
            self.decoder.weight = self.embed.weight

    def forward(self, tgt, hidden, context, attention_mask=None):
        out = self.embed(tgt)
        out, hidden = self.rnn(out, hidden)
        if self.attend:
            out, attn_weights = self.attention(
                out, context, attention_mask=attention_mask
            )
        # out = F.tanh(out)
        out = self.drop(out)
        out = self.decoder(out)
        return out, hidden

"""


class RNNDecoderCell(nn.Module):
    def __init__(
        self, hidden_size, vocab_size, embedding_size=256,
        layers=1, dropout=0.1, attention=False,
        rnn_type='gru', tie=False
    ):
        super(RNNDecoderCell, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.rnn_type = rnn_type
        rnn_cls = nn.LSTM if self.rnn_type == 'lstm' else nn.GRU
        self.rnn = rnn_cls(
            embedding_size, hidden_size, num_layers=layers, batch_first=True, bidirectional=False
        )
        self.attend = attention
        if self.attend:
            self.attention = GlobalAttention(hidden_size, hidden_size)
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_size, vocab_size)

        if tie:
            self.decoder.weight = self.embed.weight

    def forward(self, tgt, hidden, context, attention_mask=None):
        self.rnn.flatten_parameters()
        out = self.embed(tgt)
        out, hidden = self.rnn(out, hidden)
        if self.attend:
            out, attn_weights = self.attention(
                out.squeeze(), context, attention_mask=attention_mask
            )
            out = out.unsqueeze(1)
        out = self.drop(out)
        out = self.decoder(out)
        return out, hidden


class Seq2SeqRNN(nn.Module):
    def __init__(
        self, hidden_size, encoder_vocab_size, decoder_vocab_size,
        embedding_size=256, tie_decoder=False, encoder_layers=1,
        decoder_layers=1, encoder_dropout=0.1, decoder_dropout=0.1,
        rnn_type='gru', packed_sequence=True, attention=False, 
        sos=1, eos=2, teacher_forcing_p=.4,
    ):
        super(Seq2SeqRNN, self).__init__()
        self.eos = eos
        self.sos = sos
        self.out_size = decoder_vocab_size
        self.hidden_size = hidden_size
        self.teacher_forcing_p = teacher_forcing_p
        self.encoder = RNNEncoder(
            hidden_size, encoder_vocab_size, embedding_size=embedding_size,
            layers=encoder_layers, dropout=encoder_dropout,
            rnn_type=rnn_type, packed_sequence=packed_sequence
        )

        self.decoder = RNNDecoderCell(
            hidden_size, decoder_vocab_size, embedding_size=embedding_size,
            layers=decoder_layers, dropout=decoder_dropout, attention=attention,
            rnn_type=rnn_type, tie=tie_decoder
        )

    def encode(self, src, src_len):
        encoder_outputs, encoder_hidden = self.encoder(src, src_len)
        attention_mask = pad_mask(src_len, device=src.device)
        return encoder_outputs, encoder_hidden, attention_mask

    def decoder_outputs(self, tgt, encoder_hidden, encoder_outputs, attention_mask=None, teacher_forcing_p=0.):
        outputs = []
        inputs =  torch.zeros(tgt.size(0), 1, dtype=tgt.dtype) + self.sos
        inputs = inputs.to(tgt.device)
        hidden = encoder_hidden
        for i in range(tgt.size(1)):
            dec, hidden = self.decoder(
                inputs, hidden, encoder_outputs,
                attention_mask=attention_mask
            )
            teacher_force = random.random() < teacher_forcing_p
            outputs.append(dec)
            inputs = tgt[:, i] if teacher_force else dec.squeeze().argmax(1)
            inputs = inputs.unsqueeze(-1)
        outputs = torch.cat(outputs, dim=1)
        return outputs

    def forward(self, src, tgt, src_len):
        encoder_outputs, encoder_hidden, attention_mask = self.encode(src, src_len)
        teacher_forcing_p = 0. if not self.training else self.teacher_forcing_p
        decoder_outputs = self.decoder_outputs(
            tgt, encoder_hidden, encoder_outputs, attention_mask=attention_mask, teacher_forcing_p=teacher_forcing_p
        )
        return decoder_outputs

