import numpy as np
import os 

import torch
import torch.nn as nn

from ignite.metrics import Loss, Accuracy
from sklearn.preprocessing import LabelEncoder

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, ConcatDataset, SubsetRandomSampler

from slp.data.collators import DACollator
from slp.data.amz import AmazonZiser17
from slp.data.transforms import ToTokenIds, ToTensor
from slp.modules.daclassifer import DAClassifier, DALoss, DASubsetRandomSampler
from slp.modules.vat import ConditionalEntropyLoss, VADALoss, VAT, VADAWordRNN, VADAClassifier
from slp.modules.rnn import WordRNN
from slp.trainer.trainer import VADATrainer
from slp.util.embeddings import EmbeddingsLoader
from nltk.tokenize import wordpunct_tokenize

class MyTokenizer(object):
   def __init__(self, lower=True):
      self.lower = lower
   def __call__(self, x):
      if self.lower:
         x = x.lower()
      x = wordpunct_tokenize(x)
      return x

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

def transform_t(output):
    y_pred, targets, d = output
    d_pred = d['domain_pred']
    d_targets = d['domain_targets']
    y_pred = torch.stack([p for p,d in zip(y_pred, d_targets) if d==1])
    return y_pred

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

import argparse 
parser = argparse.ArgumentParser(description="Domains and losses")
parser.add_argument("-s", "--source", default="books", help="Source Domain")
parser.add_argument("-t", "--target", default="dvd", help="Target Domain")
parser.add_argument("-a", type=float, default=0.01, help="Domain Adversarial HyperParameter")
parser.add_argument("-b", type=float, default=0.01, help="C.E. HyperParameter")
parser.add_argument("-c", type=float, default=0.01, help="VAT HyperParameter")
parser.add_argument("-i", default="0", help="Path")
args = parser.parse_args()
SOURCE = args.source
TARGET = args.target
a = args.a
b = args.b
c = args.c
path = args.i

def dataloaders_from_datasets(source_dataset, target_dataset, test_dataset,
                              batch_train, batch_val, batch_test, 
                              val_size=0.2):
    dataset = ConcatDataset([source_dataset, target_dataset])

    s_dataset_size = len(source_dataset)
    s_indices = list(range(s_dataset_size))
    s_val_split = int(np.floor(val_size * s_dataset_size))
    s_train_indices = s_indices[s_val_split:]
    s_val_indices = s_indices[:s_val_split]

    t_dataset_size = len(target_dataset)
    t_indices = list(range(t_dataset_size))
    t_val_split = int(np.floor(val_size * t_dataset_size))
    t_train_indices = t_indices[t_val_split:]
    t_val_indices = t_indices[:t_val_split]

    testset_size = len(test_dataset)
    test_indices = list(range(testset_size))
    x = 16
    train_sampler = DASubsetRandomSampler(s_train_indices, t_train_indices, s_dataset_size, x, batch_train)
    val_sampler = DASubsetRandomSampler(s_val_indices, t_val_indices, s_dataset_size, x, batch_val)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_train,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=collate_fn)
    val_loader = DataLoader(
        dataset,
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


collate_fn = DACollator(device='cpu')

if __name__ == '__main__':
    loader = EmbeddingsLoader(
        './cache/glove.840B.300d.txt', 300)
    word2idx, _, embeddings = loader.load()

    tokenizer = MyTokenizer()
    #tokenizer = SpacyTokenizer()
    to_token_ids = ToTokenIds(word2idx)
    to_tensor = ToTensor(device='cpu')

    source_dataset = AmazonZiser17(ds=SOURCE, dl=0, labeled=True)
    source_dataset = source_dataset.map(tokenizer)
    source_dataset = source_dataset.map(to_token_ids)
    source_dataset = source_dataset.map(to_tensor)

    target_dataset = AmazonZiser17(ds=TARGET, dl=1, labeled=False)
    target_dataset = target_dataset.map(tokenizer)
    target_dataset = target_dataset.map(to_token_ids)
    target_dataset = target_dataset.map(to_tensor)

    test_dataset = AmazonZiser17(ds=TARGET, dl=1, labeled=True)
    test_dataset = test_dataset.map(tokenizer)
    test_dataset = test_dataset.map(to_token_ids)
    test_dataset = test_dataset.map(to_tensor)

    train_loader, dev_loader, test_loader = dataloaders_from_datasets(source_dataset, 
                                                                      target_dataset, 
                                                                      test_dataset, 
                                                                      32, 32, 1)

    sent_encoder = VADAWordRNN(256, embeddings, bidirectional=True, merge_bi='cat',
                               packed_sequence=True, attention=True, device=DEVICE)
    model = VADAClassifier(sent_encoder, 512, 2, 2)

    optimizer = Adam([p for p in model.parameters() if p.requires_grad],
                     lr=1e-3)
    cl_loss = nn.CrossEntropyLoss()
    da_loss = nn.CrossEntropyLoss()
    trg_cent_loss = ConditionalEntropyLoss()
    vat_loss = VAT(model)
    criterion = VADALoss(cl_loss, da_loss, trg_cent_loss, vat_loss, a, b, c)
    metrics = {
        'loss': Loss(criterion),
        'accuracy': Accuracy(transform_pred_tar),
        'Domain accuracy': Accuracy(transform_d),
        'Class. Loss': Loss(cl_loss, transform_pred_tar),
        'D.A. Loss': Loss(da_loss, transform_d)
    }

    trainer = VADATrainer(model, optimizer,
                      checkpoint_dir=os.path.join("./checkpoints", path),
                      metrics=metrics,
                      non_blocking=True,
                      retain_graph=True,
                      patience=5,
                      loss_fn=criterion,
                      device=DEVICE)
    trainer.fit(train_loader, dev_loader, test_loader, epochs=20)
    trainer = VADATrainer(model, optimizer=None,
                          checkpoint_dir= os.path.join("./checkpoints", path),
                          model_checkpoint='experiment_model.best.pth',
                          device=DEVICE)
    print(a, b, c, SOURCE, TARGET)
    print(evaluation(trainer, test_loader, DEVICE))
