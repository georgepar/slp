import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from slp.modules.rnn import RNN, WordRNN

class Encoder(nn.Module):
    def __init__(self, embedding, hidden_size, embeddings_dropout=.1,
                 finetune_embeddings=False, num_layers=1, batch_first=True,
                 bidirectional=False, dropout=0, attention=None, device='cpu'):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.device = device
        self.word_rnn = WordRNN(hidden_size, embedding, embeddings_dropout,
                                finetune_embeddings, batch_first,
                                num_layers, bidirectional, merge_bi='cat',
                                dropout=dropout, attention=attention,
                                device=device)

    def forward(self, inputs, lengths):
        out = self.word_rnn(inputs, lengths)

        #1. to word rnn prepei na girisei kai hidden state!!!
        #2. an einai bidirectional prepei na enwsw ta hidden states forward
        # kai backward kai episis na kanw L2 pooling over!!!
        #3.  Episis na tsekarw an kanthe fora to hidden einai 0 !!!

        return out


class ContextEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True,
                 bidirectional=False, dropout=0, attention=None,
                 rnn_type='lstm', device='cpu'):
        super(ContextEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.device = device
        self.rnn = RNN(input_size, hidden_size, batch_first, num_layers,
                       bidirectional, merge_bi='cat', dropout=dropout,
                       rnn_type=rnn_type, device=device)

    def forward(self, encoded_context):
        # to encoded_context einai ena sequence apo representations twn query.
        # se auth thn periptwsi to seq len einai 2 afou exw 2 queries!!
        #encoded_context shape: [bactchsize,seqlen,hiddensize of encoder]
        # Se auth thn periptwsi den xreiazetai pack padded seq!!!!

        out, last_hidden, hidden = self.rnn(encoded_context)

        # return last hidden!!
        return out, last_hidden, hidden


class Decoder(nn.Module):
    """
    This implementation of the decoder is only used for the referenced paper in
    HRED class. That's because of the used of some linear layers, max-out
    methods! The decoder also does not uses WordRnn class as it should
    because we wanted the embedding layer

    """
    def __init__(self, vocab_size, emb_size, hidden_size, embeddings=None,
                 embeddings_dropout=.1, finetune_embeddings=False,
                 num_layers=1, tc=1., batch_first=True, bidirectional=False,
                 dropout=0, attention=None, device='cpu'):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.teacher_forcing = tc
        self.batch_first = batch_first
        self.device = device
        self.word_rnn = RNN(hidden_size, embeddings, embeddings_dropout,
                                finetune_embeddings, batch_first,
                                num_layers, bidirectional, merge_bi='cat',
                                dropout=dropout, attention=attention,
                                packed_sequence=False,
                                device=device)

    def forward_step(self, dec_input, dec_hidden, enc_output):

        self.word_rnn(dec_input)


        return dec_out,dec_hidden

    def forward(self, dec_input, targets, target_lens, dec_hidden=None,
                enc_output=None):
        """
        dec_hidden is used for decoder's hidden state initialization!
        Usually the encoder's last (from the last timestep) hidden state is
        passed to decoder's hidden state.
        enc_output argument is passed if we want to have attention (it is
        used only for attention, if you don't want to have attention on your
        model leave it as is!)
        """
        max_seq_len = targets.shape[1]
        decoder_outputs = []

        for i in range(0, max_seq_len):
            use_teacher_forcing = True if (
                    random.random() < self.teacher_forcing_ratio) else False

            if use_teacher_forcing:
                dec_out, dec_hidden = self.forward_step(dec_input, dec_hidden,
                                                        enc_output)




            else:
                pass


        #return decoder_outputs,decoder_hidden



class HRED_MovieTriples(nn.Module):
    def __init__(self, options):
        super(HRED_MovieTriples, self).__init__()

        self.enc = Encoder()
        self.cont_enc = ContextEncoder()
        self.dec = Decoder()
        self.batch_first = options.batch_first

        #we use a linear layer and tanh act function to initialize the
        # hidden of the decoder.
        # paper reference: A Hierarchical Recurrent Encoder-Decoder
        # for Generative Context-Aware Query Suggestion, 2015
        # dm,0 = tanh(D0sm−1 + b0)  (equation 7)
        self.cont_enc_to_dec = nn.Linear(self.cont_enc.hidden_size,
                                         self.dec.hidden_size, bias=True)
        self.tanh = nn.Tanh()




    def forward(self,u1,l1,u2,l2,u3,l3):

        _,last_hidden1,_=self.enc(u1,l1)
        _, last_hidden2, _ = self.enc(u2, l2)

        # unsqueeze last hidden dim=1
        context_input = torch.cat((last_hidden1,last_hidden2),dim=1)
        # no need to use pack padded seq!!
        _,con_last_hidden,_=self.cont_enc(context_input)

        dec_init_hidden = self.tanh(self.cont_enc_to_dec(con_last_hidden))
        # edw mia view to dec_init_hidden
        # init_hidn = init_hidn.view(self.num_lyr, target.size(0), self.hid_size)

        decoder_input = torch.zeros()

        dec_out = self.dec( decoder_input, u3, l3, dec_init_hidden )
