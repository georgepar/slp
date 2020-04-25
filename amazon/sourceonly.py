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
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

collate_fn = SequenceClassificationCollator(device='cpu')

if __name__ == '__main__':
    loader = EmbeddingsLoader(
        './cache/glove.840B.300d.txt', 300)
    word2idx, _, embeddings = loader.load()

    tokenizer = SpacyTokenizer()
    to_token_ids = ToTokenIds(word2idx)
    to_tensor = ToTensor(device='cpu')

    dataset = AmazonZiser17(ds="kitchen", dl=0, labeled=True)
    dataset = dataset.map(tokenizer)
    dataset = dataset.map(to_token_ids)
    dataset = dataset.map(to_tensor)
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
        batch_size=32,
        sampler=train_sampler,
        drop_last=False,
        collate_fn=collate_fn)
    val_loader = DataLoader(
        dataset,
        batch_size=32,
        sampler=val_sampler,
        drop_last=False,
        collate_fn=collate_fn)
    
    dataset2 = AmazonZiser17(ds="electronics", dl=1, labeled=True)
    dataset2 = dataset2.map(tokenizer)
    dataset2 = dataset2.map(to_token_ids)
    dataset2 = dataset2.map(to_tensor)
    test_loader = DataLoader(
        dataset2,
        batch_size=32,
        drop_last=False,
        collate_fn=collate_fn)


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
                      patience=10,
                      loss_fn=criterion,
                      device=DEVICE)
    trainer.fit(train_loader, val_loader, test_loader, epochs=20)
