import numpy as np

import torch
import torch.nn as nn

from ignite.metrics import Loss, Accuracy
from sklearn.preprocessing import LabelEncoder

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, ConcatDataset, SubsetRandomSampler

from slp.data.collators import SequenceClassificationCollator
from slp.data.amz import AmazonZiser17
from slp.data.transforms import SpacyTokenizer, ToTokenIds, ToTensor
from slp.modules.classifier import Classifier
from slp.modules.rnn import WordRNN
from slp.trainer.trainer import SequentialTrainer
from slp.util.embeddings import EmbeddingsLoader

def transform_pred_tar(output):
    y_pred, targets, d  = output
    return y_pred, targets


def transform_d(output):
    y_pred, targets, d = output
    d_pred = d['domain_pred']
    d_targets = d['domain_targets']
    return d_pred, d_targets

#DEVICE = 'cpu'
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'

collate_fn = SequenceClassificationCollator(device='cpu')

def dataloaders_from_indices(dataset, train_indices, val_indices, batch_train, batch_val):
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_train,
        sampler=train_sampler,
        drop_last=False,
        collate_fn=collate_fn)
    val_loader = DataLoader(
        dataset,
        batch_size=batch_val,
        sampler=val_sampler,
        drop_last=False,
        collate_fn=collate_fn)
    return train_loader, val_loader


def train_test_split(dataset, batch_train, batch_val,
                     test_size=0.2, shuffle=True, seed=None):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    test_split = int(np.floor(test_size * dataset_size))
    if shuffle:
        if seed is not None:
            np.random.seed(seed)

    train_indices = indices[test_split:]
    val_indices = indices[:test_split]
    return dataloaders_from_indices(dataset, train_indices, val_indices, batch_train, batch_val)


if __name__ == '__main__':
    loader = EmbeddingsLoader(
        './cache/glove.840B.300d.txt', 300)
    word2idx, _, embeddings = loader.load()

    tokenizer = SpacyTokenizer()
    to_token_ids = ToTokenIds(word2idx)
    to_tensor = ToTensor(device='cpu')

    dataset = AmazonZiser17(ds=["kitchen", "electronics"])
    dataset = dataset.map(tokenizer)
    dataset = dataset.map(to_token_ids)
    dataset = dataset.map(to_tensor)

    train_loader, dev_loader = train_test_split(dataset, 32, 32)

    sent_encoder = WordRNN(256, embeddings, bidirectional=True, merge_bi='cat',
                           packed_sequence=True, attention=True, device=DEVICE)
    model = Classifier(sent_encoder, 512, 2)

    optimizer = Adam([p for p in model.parameters() if p.requires_grad],
                     lr=1e-3)
    criterion = nn.CrossEntropyLoss()   
    metrics = {
        'loss': Loss(criterion),
        'accuracy': Accuracy()
    }

    trainer = SequentialTrainer(model, optimizer,
                      checkpoint_dir=None,
                      metrics=metrics,
                      non_blocking=True,
                      retain_graph=True,
                      patience=5,
                      loss_fn=criterion,
                      device=DEVICE)
    trainer.fit(train_loader, dev_loader, epochs=10)