import numpy as np

import torch
import torch.nn as nn

from ignite.metrics import Loss, Accuracy
from sklearn.preprocessing import LabelEncoder

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, ConcatDataset, SubsetRandomSampler

from slp.data.collators import DACollator
from slp.data.amz import AmazonZiser17
from slp.data.transforms import SpacyTokenizer, ToTokenIds, ToTensor
from slp.modules.daclassifer import DAClassifier, DALoss, DASubsetRandomSampler
from slp.modules.vat import ConditionalEntropyLoss, VADALoss, VAT, VADAWordRNN, VADAClassifier
from slp.modules.rnn import WordRNN
from slp.trainer.trainer import VADATrainer
from slp.util.embeddings import EmbeddingsLoader

def transform_pred_tar(output):
    y_pred, targets, d  = output
    d_targets = d['domain_targets']
    y_pred = torch.stack([p for p,t in zip(y_pred, targets) if t>=0])
    targets = torch.stack([t for t in targets if t>=0])
    return y_pred, targets


def transform_d(output):
    y_pred, targets, d = output
    d_pred = d['domain_pred']
    d_targets = d['domain_targets']
    return d_pred, d_targets


#DEVICE = 'cpu'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

collate_fn = DACollator(device='cpu')

def dataloaders_from_indices(source_dataset, target_dataset, test_dataset,
                            s_train_indices, val_indices, t_indices,
                            batch_train, batch_val):
    t_dataset_size = len(target_dataset)
    s_dataset_size = len(source_dataset)
    target_indices = list(range(t_dataset_size))
    dataset = ConcatDataset([source_dataset, target_dataset])
    x = 8
    train_sampler = DASubsetRandomSampler(s_train_indices, target_indices, s_dataset_size, x, batch_train)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(t_indices)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_train,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=collate_fn)
    val_loader = DataLoader(
        source_dataset,
        batch_size=batch_val,
        sampler=val_sampler,
        drop_last=True,
        collate_fn=collate_fn)
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        sampler=test_sampler,
        drop_last=False,
        collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


def train_test_split(source_dataset, target_dataset, test_dataset,
                     batch_train, batch_val, val_size=0.2):
    s_dataset_size = len(source_dataset)
    s_indices = list(range(s_dataset_size))
    test_dataset_size = len(test_dataset)
    t_indices = list(range(test_dataset_size))
    val_split = int(np.floor(val_size * s_dataset_size))
    train_indices = s_indices[val_split:]
    val_indices = s_indices[:val_split]
    return dataloaders_from_indices(source_dataset, target_dataset, test_dataset, train_indices, val_indices, t_indices, batch_train, batch_val)


if __name__ == '__main__':
    loader = EmbeddingsLoader(
        './cache/glove.840B.300d.txt', 300)
    word2idx, _, embeddings = loader.load()

    tokenizer = SpacyTokenizer()
    to_token_ids = ToTokenIds(word2idx)
    to_tensor = ToTensor(device='cpu')

    source_dataset = AmazonZiser17(ds="electronics", dl=0, labeled=True)
    source_dataset = source_dataset.map(tokenizer)
    source_dataset = source_dataset.map(to_token_ids)
    source_dataset = source_dataset.map(to_tensor)

    target_dataset = AmazonZiser17(ds="dvd", dl=1, labeled=False)
    target_dataset = target_dataset.map(tokenizer)
    target_dataset = target_dataset.map(to_token_ids)
    target_dataset = target_dataset.map(to_tensor)

    test_dataset = AmazonZiser17(ds="dvd", dl=1, labeled=True)
    test_dataset = test_dataset.map(tokenizer)
    test_dataset = test_dataset.map(to_token_ids)
    test_dataset = test_dataset.map(to_tensor)

    train_loader, dev_loader, test_loader = train_test_split(source_dataset, target_dataset,
                                                             test_dataset, 16, 16)

    sent_encoder = VADAWordRNN(256, embeddings, bidirectional=True, merge_bi='cat',
                               packed_sequence=True, attention=True, device=DEVICE)
    model = VADAClassifier(sent_encoder, 512, 2, 2)

    optimizer = Adam([p for p in model.parameters() if p.requires_grad],
                     lr=1e-3)
    cl_loss = nn.CrossEntropyLoss()
    da_loss = nn.CrossEntropyLoss()
    trg_cent_loss = ConditionalEntropyLoss()
    vat_loss = VAT(model)
    criterion = VADALoss(cl_loss, da_loss, trg_cent_loss, vat_loss)
    metrics = {
        'loss': Loss(criterion),
        'accuracy': Accuracy(transform_pred_tar),
        'Domain accuracy': Accuracy(transform_d),
        'Class. Loss': Loss(cl_loss, transform_pred_tar),
        'D.A. Loss': Loss(da_loss, transform_d)
    }

    trainer = VADATrainer(model, optimizer,
                      checkpoint_dir=None,
                      metrics=metrics,
                      non_blocking=True,
                      retain_graph=True,
                      patience=20,
                      loss_fn=criterion,
                      device=DEVICE)
    trainer.fit(train_loader, dev_loader, test_loader, epochs=20)
