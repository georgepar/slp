import torch
import torch.nn as nn

from slp.modules.regularization import GaussianNoise


class PositionalEncoding(nn.Module):
    """
    PositionalEncoding
    PE(pos,2i)=sin(pos/10000^(2i/dmodel))
    PE(pos,2i+1)=cos(pos/10000^(2i/dmodel))
    """
    def __init__(self, max_length, embedding_dim=512, device='cpu'):
        super(PositionalEncoding, self).__init__()
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        pe = torch.zeros(max_length, embedding_dim,
                         dtype=torch.float, device=device)
        embedding_indices = torch.arange(0, embedding_dim,
                                         dtype=torch.float, device=device)
        position_indices = (torch
                            .arange(0, max_length,
                                    dtype=torch.float, device=device)
                            .unsqueeze(-1))
        # freq => (E,)
        freq_term = 10000 ** (2 * embedding_indices / embedding_dim)
        pe[:, 0::2] = torch.sin(position_indices / freq_term[0::2])
        pe[:, 1::2] = torch.cos(position_indices / freq_term[1::2])
        # pe => (1, max_length, E)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x => (B, L, E) sequence of embedded tokens
        """
        # (B, L, E)
        return x + self.pe[:, :x.size(1)]


class Embed(nn.Module):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 embeddings=None,
                 noise=.0,
                 dropout=.0,
                 trainable=False):
        """
        Define the layer of the model and perform the initializations
        of the layers (wherever it is necessary)
        Args:
            embeddings (numpy.ndarray): the 2D ndarray with the word vectors
            noise (float):
            dropout (float):
            trainable (bool):
        """
        super(Embed, self).__init__()

        # define the embedding layer, with the corresponding dimensions
        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_dim)

        if embeddings is not None:
            print("Initializing Embedding layer with pre-trained weights!")
            self.init_embeddings(embeddings, trainable)

        # the dropout "layer" for the word embeddings
        self.dropout = nn.Dropout(dropout)

        # the gaussian noise "layer" for the word embeddings
        self.noise = GaussianNoise(noise)

    def init_embeddings(self, weights, trainable):
        self.embedding.weight = nn.Parameter(torch.from_numpy(weights),
                                             requires_grad=trainable)

    def forward(self, x):
        """
        This is the heart of the model. This function, defines how the data
        passes through the network.
        Args:
            x (): the input data (the sentences)
        Returns: the logits for each class
        """
        embeddings = self.embedding(x)

        if self.noise.stddev > 0:
            embeddings = self.noise(embeddings)

        if self.dropout.p > 0:
            embeddings = self.dropout(embeddings)

        return embeddings
