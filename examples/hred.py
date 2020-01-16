import numpy as np
import torch
import os
import torch.nn as nn

from ignite.metrics import Loss

from torch.optim import Adam

from slp.config.special_tokens import HRED_SPECIAL_TOKENS
from slp.data.utils import train_test_split
from slp.data.transforms import SpacyTokenizer, ToTokenIds, ToTensor
from slp.data.DummymovieTriples import DummyMovieTriples
from slp.data.collators import HRED_MovieTriples_Collator
from slp.util.embeddings import EmbeddingsLoader
from slp.trainer.trainer import HREDMovieTriplesTrainer
from slp.modules.loss import SequenceCrossEntropyLoss

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
COLLATE_FN = HRED_MovieTriples_Collator(device='cpu')
MAX_EPOCHS = 10
BATCH_TRAIN_SIZE = 32
BATCH_VAL_SIZE = 32


def trainer_factory(embeddings, pad_index, sos_index, device=DEVICE):

    model=1

    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=1e-6)

    criterion = SequenceCrossEntropyLoss(pad_index)

    metrics = {
        'loss': Loss(criterion)
    }

    trainer = HREDMovieTriplesTrainer(model, optimizer, checkpoint_dir=None,
                                      metrics=metrics, non_blocking=True,
                                      retain_graph=False, patience=5,
                                      device=device, loss_fn=criterion)
    return trainer


if __name__ == '__main__':

    emb_file = './cache/glove.6B.50d.txt'
    emb_dim = 300
    word2idx, idx2word, embeddings = EmbeddingsLoader(emb_file, emb_dim,
                                                      extra_tokens=
                                                      HRED_SPECIAL_TOKENS
                                                      ).load()

    tokenizer = SpacyTokenizer()
    to_token_ids = ToTokenIds(word2idx)
    to_tensor = ToTensor()
    dataset = DummyMovieTriples('./data/dummy_movie_triples', transforms=[
        tokenizer, to_token_ids, to_tensor])

    train_loader, val_loader = train_test_split(dataset, BATCH_TRAIN_SIZE,
                                                BATCH_VAL_SIZE, COLLATE_FN)

    pad_index = word2idx[HRED_SPECIAL_TOKENS.PAD.value]
    sos_index = word2idx[HRED_SPECIAL_TOKENS.SOU.value]
    eos_index = word2idx[HRED_SPECIAL_TOKENS.EOU.value]

    trainer = trainer_factory(embeddings, pad_index, sos_index, device=DEVICE)
