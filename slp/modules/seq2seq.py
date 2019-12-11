import random

import torch
import torch.nn as nn
import torch.nn.functional as f
from slp.modules.attention import Attention
from slp.modules.util import pad_mask


class EncoderLSTM(nn.Module):

    def __init__(self, weights_matrix, hidden_size, num_layers=1,
                 dropout=0, bidirectional=False, rnn_type='lstm',
                 batch_first=True, emb_train=False, device='cpu'):
        super(EncoderLSTM, self).__init__()
        self.vocab_size, self.input_size = weights_matrix.shape
        self.emb_train = emb_train
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.dropout = (0 if self.num_layers == 1 else dropout)
        self.batch_first = batch_first
        self.rnn_type = rnn_type
        self.device = device
        self.dropout_out = nn.Dropout(dropout)
        # self.embedding = self.create_emb_layer(weights_matrix,
        #                                        trainable=self.emb_train)
        self.create_emb_layer(weights_matrix, trainable=self.emb_train)
        if rnn_type == 'lstm':
            self.encoder = nn.LSTM(input_size=self.input_size,
                                   hidden_size=self.hidden_size,
                                   num_layers=self.num_layers,
                                   bidirectional=self.bidirectional,
                                   dropout=self.dropout,
                                   batch_first=self.batch_first)
        elif rnn_type == 'rnn':
            self.encoder = nn.RNN(input_size=self.input_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  bidirectional=self.bidirectional,
                                  dropout=self.dropout,
                                  batch_first=self.batch_first)
        elif rnn_type == 'gru':
            self.encoder = nn.GRU(input_size=self.input_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  bidirectional=self.bidirectional,
                                  dropout=self.dropout,
                                  batch_first=self.batch_first)

    def create_emb_layer(self, weights_matrix, trainable):
        self.embedding = nn.Embedding(self.vocab_size, self.input_size)
        weights = torch.FloatTensor(weights_matrix)

        self.embedding.weight.data.copy_(weights)
        self.embedding.weight = nn.Parameter(weights)
        self.embedding.weight.requires_grad = trainable
        #return embedding

    def forward(self, input_seq, input_lengths):
        embedded_seq = self.dropout_out(self.embedding(input_seq))
        # hidden = self._init_hidden()
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded_seq,
                                                         input_lengths,
                                                         batch_first=self.
                                                         batch_first,
                                                         enforce_sorted=False)
        if self.rnn_type == 'lstm':
            enc_out, enc_hidden = self.encoder(packed)
        else:
            enc_out, enc_hidden = self.encoder(packed)
        enc_out, _ = torch.nn.utils.rnn.pad_packed_sequence(enc_out,
                                                            batch_first=self.
                                                            batch_first)

        if self.bidirectional:
            enc_out = enc_out[:, :, :self.hidden_size] + \
                         enc_out[:, :, self.hidden_size:]

        return enc_out, enc_hidden


class DecoderLSTMv2(nn.Module):
    def __init__(self, weights_matrix, hidden_size, output_size,
                 max_target_len, emb_layer=None, num_layers=1, dropout=0,
                 bidirectional=False, batch_first=True, emb_train=False,
                 rnn_type='lstm', device='cpu'):

        super(DecoderLSTMv2, self).__init__()

        if weights_matrix is not None and emb_layer is None:
            self.vocab_size, self.input_size = weights_matrix.shape
            # self.embedding = self.create_emb_layer(weights_matrix,
            #                                        trainable=emb_train)
            self.create_emb_layer(weights_matrix, trainable=emb_train)
        elif emb_layer is not None and weights_matrix is None:
            self.embedding = emb_layer
            self.vocab_size, self.input_size = (emb_layer.num_embeddings,
                                                emb_layer.embedding_dim)
        else:
            assert False,"emb_layer and weights_matrix should not be both " \
                         "None or initialized."
        self.emb_train = emb_train
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.dropout = (0 if self.num_layers == 1 else dropout)
        self.max_target_len = max_target_len
        self.rnn_type = rnn_type
        self.device = device

        if rnn_type == 'lstm':
            self.decoder = nn.LSTM(input_size=self.input_size,
                                   hidden_size=self.hidden_size,
                                   num_layers=self.num_layers,
                                   bidirectional=self.bidirectional,
                                   dropout=self.dropout,
                                   batch_first=self.batch_first)
        elif rnn_type == 'rnn':
            self.decoder = nn.RNN(input_size=self.input_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  bidirectional=self.bidirectional,
                                  dropout=self.dropout,
                                  batch_first=self.batch_first)
        elif rnn_type == 'gru':
            self.decoder = nn.GRU(input_size=self.input_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  bidirectional=self.bidirectional,
                                  dropout=self.dropout,
                                  batch_first=self.batch_first)

        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.dropout_out = nn.Dropout(dropout)

    def create_emb_layer(self, weights_matrix, trainable):
        self.embedding = nn.Embedding(self.vocab_size, self.input_size)
        weights = torch.FloatTensor(weights_matrix)
        # embedding.weight.data.copy_(weights)
        self.embedding.weight = nn.Parameter(weights)
        self.embedding.weight.requires_grad = trainable

        #return embedding

    def forward(self, dec_input, dec_hidden):

        dec_input = dec_input.long()
        dec_input.to(self.device)
        embedded = self.dropout_out(self.embedding(dec_input))

        decoder_output, decoder_hidden = self.decoder(embedded,
                                                      dec_hidden)
        if self.bidirectional:
            decoder_output = decoder_output[:, :, :self.hidden_size] + \
                             decoder_output[:, :, self.hidden_size:]
        decoder_output = self.out(decoder_output)
        return decoder_output, decoder_hidden


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, bos_indx,
                 teacher_forcing_ratio=0, device='cpu'):
        super(EncoderDecoder, self).__init__()

        # initialize the encoder and decoder
        self.bos_indx = bos_indx
        self.encoder = encoder
        self.decoder = decoder
        self.max_target_len = self.decoder.max_target_len
        self.teacher_forcing_ratio = teacher_forcing_ratio
        # index for the decoder!
        self.device = device

    def forward(self, input_seq, lengths_inputs, target_seq):
        batch_size = input_seq.shape[0]

        encoder_output, encoder_hidden = self.encoder(input_seq,
                                                      lengths_inputs)
        decoder_input = [[self.bos_indx for _ in range(
            batch_size)]]

        decoder_input = torch.tensor(decoder_input).long()
        decoder_input = decoder_input.transpose(0, 1)
        decoder_input = decoder_input.to(self.device)

        if self.encoder.rnn_type == "lstm":
            decoder_hidden = (encoder_hidden[0][:self.decoder.num_layers],
                              encoder_hidden[1][:self.decoder.num_layers])
        else:
            decoder_hidden = encoder_hidden[:self.decoder.num_layers]

        # Determine if we are using teacher forcing this iteration
        use_teacher_forcing = True if random.random() < self. \
            teacher_forcing_ratio else False
        decoder_all_outputs = []
        if use_teacher_forcing:

            for t in range(0, target_seq.shape[1]):
                decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                              decoder_hidden)
                decoder_all_outputs.append(
                    torch.squeeze(decoder_output, dim=1))
                # Teacher forcing: next input is current target

                decoder_input = target_seq[:, t]
                decoder_input = torch.unsqueeze(decoder_input, dim=1)

        else:

            for t in range(0, target_seq.shape[1]):
                decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                              decoder_hidden)
                decoder_all_outputs.append(
                    torch.squeeze(decoder_output, dim=1))
                current_output = torch.squeeze(decoder_output, dim=1)
                top_index = f.log_softmax(current_output, dim=0)
                value, pos_index = top_index.max(dim=1)
                # value, pos_index = current_output.max(dim=1)
                decoder_input = torch.unsqueeze(pos_index, dim=1)
                decoder_input = decoder_input.to(self.device)

        decoder_all_outputs = torch.stack(decoder_all_outputs).transpose(0, 1)

        return decoder_all_outputs

    def evaluate(self, input_seq, eos_index):
        """
        This function is only used for live-interaction with model!
        It was created to be used for input interaction with the model!
        """

        input_seq = torch.unsqueeze(input_seq, dim=0)
        input_seq = input_seq.to(self.device)
        input_len = torch.tensor([len(input_seq)])

        encoder_out, encoder_hidden = self.encoder(input_seq, input_len)
        decoder_input = [[self.bos_indx]]
        decoder_input = torch.tensor(decoder_input).long()
        decoder_input = decoder_input.transpose(0, 1)
        decoder_input = decoder_input.to(self.device)
        if self.encoder.rnn_type == "lstm":
            decoder_hidden = (encoder_hidden[0][:self.decoder.num_layers],
                              encoder_hidden[1][:self.decoder.num_layers])
        else:
            decoder_hidden = encoder_hidden[:self.decoder.num_layers]

        decoder_all_outputs = []

        for t in range(0, self.max_target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                          decoder_hidden)

            current_output = torch.squeeze(decoder_output, dim=1)
            top_index = f.log_softmax(current_output, dim=0)
            value, pos_index = top_index.max(dim=1)
            # value, pos_index = current_output.max(dim=1)
            if pos_index == eos_index:
                break
            decoder_all_outputs.append(pos_index)
            decoder_input = torch.unsqueeze(pos_index, dim=1)

        return decoder_all_outputs


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout),
                          bidirectional=True,batch_first=True)

    def forward(self, input_seq, input_lengths, hidden=None):

        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module

        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths,
                                                   batch_first=True,
                                                   enforce_sorted=False)

        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs,
                                                      batch_first=True)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :,
                                                     self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden


# Luong attention layer
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method,
                             "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        #import ipdb;ipdb.set_trace()
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat(
            (hidden.expand(encoder_output.size(0), -1, -1), encoder_output),
            2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        #attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return f.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          batch_first=True, dropout=(0 if n_layers == 1 else
                                                     dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        #import ipdb;ipdb.set_trace()
        context = attn_weights.bmm(encoder_outputs)
        #import ipdb;ipdb.set_trace()
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(1)
        context = context.squeeze(1)
        #context = context.transpose(0,1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = f.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden
