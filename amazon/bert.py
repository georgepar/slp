import numpy as np

import torch
import torch.nn as nn

from ignite.metrics import Loss, Accuracy
from sklearn.preprocessing import LabelEncoder

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, ConcatDataset, SubsetRandomSampler

from slp.data.collators import BertCollator
from transformers import *
from slp.data.bertamz import AmazonZiser17
from slp.data.transforms import SpacyTokenizer, ToTokenIds, ToTensor
from slp.modules.classifier import Classifier
from slp.modules.rnn import WordRNN
from slp.trainer.trainer import BertTrainer
from slp.util.embeddings import EmbeddingsLoader

import argparse
parser = argparse.ArgumentParser(description="Domains and losses")
parser.add_argument("-s", "--source", default="books", help="Source Domain")
parser.add_argument("-t", "--target", default="dvd", help="Target Domain")
args = parser.parse_args()
SOURCE = args.source
TARGET = args.target

def transform_pred_tar(output):
    y_pred, targets, d  = output
    return y_pred, targets

def transform_d(output):
    y_pred, targets, d = output
    d_pred = d['domain_pred']
    d_targets = d['domain_targets']
    return d_pred, d_targets

def evaluation(trainer, test_loader, device):
    trainer.model.eval()
    predictions = []
    labels = []
    metric = Accuracy()
    with torch.no_grad():
        for index, batch in enumerate(test_loader):
            review = batch[0].to(device)
            label = batch[1].to(device)
            length = batch[2].to(device)
            pred = trainer.model(review, length)
            metric.update((pred, label))
    acc = metric.compute()
    return acc

#DEVICE = 'cpu'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

collate_fn = BertCollator(device='cpu')

if __name__ == '__main__':
    dataset = AmazonZiser17(ds=SOURCE, dl=0, labeled=True)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    val_size = 0.2
    val_split = int(np.floor(val_size * dataset_size))
    train_indices = indices[val_split:]
    val_indices = indices[:val_split]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=train_sampler,
        drop_last=False,
        collate_fn=collate_fn)
    val_loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=val_sampler,
        drop_last=False,
        collate_fn=collate_fn)

    dataset2 = AmazonZiser17(ds=TARGET, dl=1, labeled=True)
    test_loader = DataLoader(
        dataset2,
        batch_size=1,
        drop_last=False,
        collate_fn=collate_fn)


    bertmodel = BertModel.from_pretrained('bert-base-uncased')
    model = Classifier(bertmodel, 768, 2)

    optimizer = Adam([p for p in model.parameters() if p.requires_grad],
                     lr=1e-3)
    criterion = nn.CrossEntropyLoss() 
    metrics = {
        'loss': Loss(criterion),
        'accuracy': Accuracy()
    }

    trainer = BertTrainer(model, optimizer,
                      newbob_period=3,
                      checkpoint_dir='./checkpoints/',
                      metrics=metrics,
                      non_blocking=True,
                      retain_graph=True,
                      patience=5,
                      loss_fn=criterion,
                      device=DEVICE)
    trainer.fit(train_loader, val_loader, epochs=20)
    trainer = BertTrainer(model, optimizer=None,
                      checkpoint_dir='./checkpoints/',
                      model_checkpoint='experiment_model.best.pth',
                      device=DEVICE)
    print(evaluation(trainer, test_loader, DEVICE))
