import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class EncoderLSTM(nn.Module):

    def __init__(self, weights_matrix, hidden_size,  num_layers=1,
                 dropout=0, bidirectional=False, batch_first=False,
                 device='cpu'):
        super(EncoderLSTM, self).__init__()
        self.vocab_size, self.input_size = weights_matrix.shape
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.dropout = (0 if self.num_layers == 1 else dropout)
        self.batch_first = batch_first
        self.device = device

        if self.bidirectional:
            # Bidirectional has twice the amount of hidden variables so if you
            # wan’t to keep the final output the same you have to divide the
            # hidden_dim by 2
            self.embedding = self.create_emb_layer(weights_matrix,
                                                   trainable=False)
            self.encoder = nn.LSTM(input_size=self.input_size,
                                   hidden_size=self.hidden_size//2,
                                   num_layers=self.num_layers,
                                   bidirectional=self.bidirectional,
                                   dropout=self.dropout,
                                   batch_first=self.batch_first)
        else:
            self.embedding = self.create_emb_layer(weights_matrix,
                                                   trainable=False)
            self.encoder = nn.LSTM(input_size=self.input_size,
                                   hidden_size=self.hidden_size,
                                   num_layers=self.num_layers,
                                   bidirectional=self.bidirectional,
                                   dropout=self.dropout,
                                   batch_first=self.batch_first)

    def create_emb_layer(self, weights_matrix, trainable=False):
        embedding = nn.Embedding(self.vocab_size, self.input_size)
        weights = torch.FloatTensor(weights_matrix)
        embedding.weight.data.copy_(weights)
        embedding.weight = nn.Parameter(weights)
        embedding.weight.requires_grad = trainable
        return embedding

    def _init_hidden(self):
        return None

    def forward(self, input_seq, input_lengths):
        embedded_seq = self.embedding(input_seq)
        hidden = self._init_hidden()
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded_seq,
                                                         input_lengths,
                                                         batch_first=
                                                         self.batch_first,
                                                         enforce_sorted=False)

        enc_out, (enc_hidden, enc_cell) = self.encoder(packed, hx=hidden)
        enc_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            enc_out, batch_first=self.batch_first)

        if self.bidirectional:
            # if we have a bidirectional encoder we concatenate hidden states
            # (forward and backward)
            # the first dimension in enc_hidden[0] is the number of layers,
            # the second is the number of batches and the third is the
            # hidden size div 2 (for bidirectional). So we have to take the
            # forward hidden state of each layers.
            # pytorch returns hidden state in the form described above in
            # this situation. The first index (index 0) is the forward
            # first layer. The second (index 1) is the backward of first
            # layer.etc. If we have num_layers=3 the hidden state will be in
            # the form [6,num_batches , hidden_size//2]
            # so now we will receive the forward hidden states (using step 2)
            # of each layers and then the backward

            # forward_hidden = enc_hidden[0:self.num_layers:2]
            # backward_hidden = enc_hidden[1:self.num_layers:2]

            forward_hidden = enc_hidden[0:2*self.num_layers:2]
            backward_hidden = enc_hidden[1:2*self.num_layers:2]
            forward_cell = enc_cell[0:2*self.num_layers:2]
            backward_cell = enc_cell[1:2*self.num_layers:2]

            new_hidden = torch.cat([forward_hidden, backward_hidden], dim=2)
            new_cell = torch.cat([forward_cell, backward_cell], dim=2)

            return enc_out, (new_hidden, new_cell)
        else:
            return enc_out, (enc_hidden, enc_cell)


class DecoderLSTM_v2(nn.Module):
    def __init__(self, weights_matrix, hidden_size, output_size,
                 max_target_len, num_layers=1, dropout=0, bidirectional=False,
                 batch_first=False,device='cpu'):

        super(DecoderLSTM_v2, self).__init__()
        self.vocab_size, self.input_size = weights_matrix.shape
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.dropout = (0 if self.num_layers == 1 else dropout)
        self.max_target_len = max_target_len
        self.device = device

        if self.bidirectional:
            self.embedding = self.create_emb_layer(weights_matrix,
                                                   trainable=False)
            self.decoder = nn.LSTM(input_size=self.input_size,
                                   hidden_size=self.hidden_size//2,
                                   num_layers=self.num_layers,
                                   bidirectional=self.bidirectional,
                                   dropout=self.dropout,
                                   batch_first=self.batch_first)

            self.out = nn.Linear(self.hidden_size//2, self.output_size)

        else:
            self.embedding = self.create_emb_layer(weights_matrix,
                                                   trainable=False)
            self.decoder = nn.LSTM(input_size=self.input_size,
                                   hidden_size=self.hidden_size,
                                   num_layers=self.num_layers,
                                   bidirectional=self.bidirectional,
                                   dropout=self.dropout,
                                   batch_first=self.batch_first)

            self.out = nn.Linear(self.hidden_size, self.output_size)

    def create_emb_layer(self, weights_matrix, trainable=False):
        embedding = nn.Embedding(self.vocab_size, self.input_size)
        weights = torch.FloatTensor(weights_matrix)
        #embedding.weight.data.copy_(weights)
        embedding.weight = nn.Parameter(weights)
        embedding.weight.requires_grad = trainable

        return embedding

    def forward(self, dec_input, dec_hidden):
        dec_input.to(self.device)
        embedded = self.embedding(dec_input)
        decoder_output, decoder_hidden = self.decoder(embedded,
                                                      dec_hidden)
        decoder_output = self.out(decoder_output)
        return decoder_output, decoder_hidden


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, bos_indx,
                 teacher_forcing_ratio=0,device='cpu'):
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
        encoder_output, encoder_hidden = self.encoder(input_seq, lengths_inputs)
        decoder_input = [[self.bos_indx for _ in range(
            batch_size)]]
        decoder_input = torch.LongTensor(decoder_input)
        decoder_input = decoder_input.transpose(0, 1)
        decoder_hidden = encoder_hidden[:self.decoder.num_layers]
        decoder_input = decoder_input.to(self.device)
        # Determine if we are using teacher forcing this iteration
        use_teacher_forcing = True if random.random() \
                                      < self.teacher_forcing_ratio else False
        decoder_all_outputs = []
        if use_teacher_forcing:
            for t in range(0, self.max_target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                              decoder_hidden)
                decoder_all_outputs.append(torch.squeeze(decoder_output, dim=1))
                # Teacher forcing: next input is current target
                decoder_input = target_seq[:, t]
                decoder_input = torch.unsqueeze(decoder_input, dim=1)

        else:
            for t in range(0, self.max_target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                              decoder_hidden)
                decoder_all_outputs.append(torch.squeeze(decoder_output, dim=1))
                current_output = torch.squeeze(decoder_output, dim=1)
                top_index = F.log_softmax(current_output, dim=1)
                value, pos_index = top_index.max(dim=1)
                decoder_input = [index for index in pos_index]
                decoder_input = torch.LongTensor(decoder_input)
                decoder_input = torch.unsqueeze(decoder_input,dim=1)
                decoder_input = decoder_input.to(self.device)

        return decoder_all_outputs

    def evaluate(self, input_seq, lengths_inputs):
        batch_size = input_seq.shape[0]
        encoder_output, encoder_hidden = self.encoder(input_seq, lengths_inputs)
        decoder_input = [[self.bos_indx for _ in range(
            batch_size)]]
        decoder_input = torch.LongTensor(decoder_input)
        decoder_input = decoder_input.transpose(0, 1)
        decoder_hidden = encoder_hidden[:self.decoder.num_layers]
        decoder_input = decoder_input.to(self.device)
        decoder_all_outputs = []

        for t in range(0, self.max_target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                          decoder_hidden)
            decoder_all_outputs.append(torch.squeeze(decoder_output,
                                                     dim=1).tolist())
            current_output = torch.squeeze(decoder_output, dim=1)
            top_index = F.softmax(current_output, dim=0)
            value, pos_index = top_index.max(dim=1)
            decoder_input = [index for index in pos_index]
            decoder_input = torch.LongTensor(decoder_input)
            decoder_input = torch.unsqueeze(decoder_input,dim=1)
            decoder_input = decoder_input.to(self.device)
        decoder_all_outputs = (torch
                               .FloatTensor(decoder_all_outputs)
                               .transpose(0,1))
        return decoder_all_outputs, decoder_hidden
