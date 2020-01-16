import numpy as np
import torch
import os
import torch.nn as nn

from ignite.metrics import Loss

from torch.optim import Adam
from torchvision.transforms import Compose

from slp.data.utils import train_test_split
from slp.data.transforms import SpacyTokenizer, ToTokenIds, ToTensor
from slp.data.DummymovieTriples import DummyMovieTriples
from slp.data.collators import HRED_MovieTriples_Collator
from slp.util.embeddings import EmbeddingsLoader
from slp.trainer.trainer import HREDMovieTriplesTrainer


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
COLLATE_FN = HRED_MovieTriples_Collator(device='cpu')
MAX_EPOCHS = 10
BATCH_TRAIN_SIZE = 32
BATCH_VAL_SIZE = 32


if __name__ == '__main__':

    emb_file = './cache/glove.6B.50d.txt'

    dataset = DummyMovieTriples('./data/', transforms=None)

