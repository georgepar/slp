import torch
import torch.nn as nn
from slp.modules.embed import Embed
from slp.modules.rnn import RNN

class MyRNN(nn.Module):
    def __init__(self, embeddings, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        
        super().__init__()
        vocab_size = embeddings.shape[0]
        embedding_dim = embeddings.shape[1]
        '''
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        #self.embedding.weight.data.copy_(torch.from_numpy(embeddings))
        #self.embedding.weight.requires_grad = False
        self.embedding.weight = nn.Parameter(torch.from_numpy(embeddings),
                                             requires_grad=False)
        '''
        self.embedding = Embed(embeddings.shape[0],
                           embeddings.shape[1],
                           embeddings=embeddings,
                           dropout=dropout,
                           trainable=False)
 
        #self.rnn = nn.LSTM(embedding_dim, 
        #                   hidden_dim, 
        #                   num_layers=n_layers, 
        #                   bidirectional=bidirectional, 
        #                   dropout=dropout)

        self.rnn = RNN(embedding_dim, hidden_dim, bidirectional=bidirectional, dropout=dropout)        

        #self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        #text = text.transpose(0, 1)  
        #text = [sent len, batch size]
        
        embedded = self.dropout(self.embedding(text))
        
        #embedded = [sent len, batch size, emb dim]
        
        #pack sequence
        #packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, enforce_sorted=False)
        
        #packed_output, (hidden, cell) = self.rnn(packed_embedded)
        
        #unpack sequence
        #output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        output, hidden = self.rnn(embedded, text_lengths)

        #output = [sent len, batch size, hid dim * num directions]
        #output over padding tokens are zero tensors
        
        #hidden = [num layers * num directions, batch size, hid dim]
        #cell = [num layers * num directions, batch size, hid dim]
        
        #concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        #and apply dropout
        
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
                
        #hidden = [batch size, hid dim * num directions]
        return hidden   
        #return self.fc(hidden)

