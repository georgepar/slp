import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
from torch.optim import Adam

from ignite.metrics import Loss, Accuracy
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.transforms import Compose
from sklearn.model_selection import KFold

from slp.data.collators_title import SequenceClassificationCollator
from slp.data.therapy_title_on2 import PsychologicalDataset, TupleDataset
from slp.data.transforms import SpacyTokenizer, ToTokenIds, ToTensor, ReplaceUnknownToken
from slp.modules.hier_att_net_title_attentional_embed import HierAttNet
from slp.util.embeddings import EmbeddingsLoader
from slp.trainer.trainer_title_no_validation import SequentialTrainer

#DEVICE = 'cpu'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

COLLATE_FN = SequenceClassificationCollator(device=DEVICE)

DEBUG = False
KFOLD = True
MAX_EPOCHS = 50

def dataloaders_from_indices(dataset, train_indices, val_indices, batch_train, batch_val):
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_train,
        sampler=train_sampler,
        drop_last=False,
        collate_fn=COLLATE_FN)
    val_loader = DataLoader(
        dataset,
        batch_size=batch_val,
        sampler=val_sampler,
        drop_last=False,
        collate_fn=COLLATE_FN)

    return train_loader, val_loader

def train_test_split(dataset, batch_train, batch_val,
                     test_size=0.1, shuffle=True, seed=42):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    test_split = int(np.floor(test_size * dataset_size))
    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)

    train_indices = indices[test_split:]
    val_indices = indices[:test_split]

    return dataloaders_from_indices(dataset, train_indices, val_indices, batch_train, batch_val)


def kfold_split(dataset, batch_train, batch_val, k=5, shuffle=True, seed=None):
    kfold = KFold(n_splits=k, shuffle=shuffle, random_state=seed)
    for train_indices, val_indices in kfold.split(dataset):
        yield dataloaders_from_indices(dataset, train_indices, val_indices, batch_train, batch_val)

def trainer_factory(embeddings, idx2word, lex_size, device=DEVICE):
    model = HierAttNet(
        hidden_size, batch_size, num_classes, max_sent_length, len(embeddings), embeddings, idx2word, lex_size)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.0005)

    metrics = {
        'accuracy': Accuracy(),
        'loss': Loss(criterion)
    }

    trainer = SequentialTrainer(
        model,
        optimizer,
        checkpoint_dir='../checkpoints' if not DEBUG else None,
        metrics=metrics,
        non_blocking=True,
        patience=10,
        loss_fn=criterion,
        device=DEVICE)

    return trainer


if __name__ == '__main__':

    ####### Parameters ########
    batch_train = 8
    batch_val = 8

    max_sent_length = 500  #max number of sentences (turns) in transcript - after padding
    max_word_length = 150   #max length of each sentence (turn) - after padding
    num_classes = 2
    batch_size = 8
    hidden_size = 300
    lex_size = 99

    epochs = 40

#    loader = EmbeddingsLoader('../data/glove.6B.300d.txt', 300)
    loader = EmbeddingsLoader('/data/embeddings/glove.840B.300d.txt', 300)
    word2idx, idx2word, embeddings = loader.load()
    embeddings = torch.tensor(embeddings)

    tokenizer = SpacyTokenizer()
    replace_unknowns = ReplaceUnknownToken()
    to_token_ids = ToTokenIds(word2idx)
    to_tensor = ToTensor(device=DEVICE)

    bio = PsychologicalDataset(
        '../data/balanced_new_csv.csv', '../../../test_CEL/slp/data/psychotherapy/',
        max_word_length,
        text_transforms = Compose([
            tokenizer,
            replace_unknowns,
            to_token_ids,
            to_tensor]))


    if KFOLD:
        cv_scores = []
        import gc
        for train_loader, val_loader in kfold_split(bio, batch_train, batch_val):
            trainer = trainer_factory(embeddings, idx2word, lex_size, device=DEVICE)
            fold_score = trainer.fit(train_loader, val_loader, epochs=MAX_EPOCHS)
            cv_scores.append(fold_score)
            print("**********************")
            print("edw")
            print(fold_score)
            del trainer
            gc.collect()
        final_score = float(sum(cv_scores)) / len(cv_scores)
    else:
        train_loader, val_loader = train_test_split(bio, batch_train, batch_val)
        trainer = trainer_factory(embeddings, idx2word, lex_size, device=DEVICE)
        final_score = trainer.fit(train_loader, val_loader, epochs=MAX_EPOCHS)

    print(f'Final score: {final_score}')




    if DEBUG:
        print("Starting end to end test")
        print("-----------------------------------------------------------------------")
        trainer.fit_debug(train_loader, val_loader)
        print("Overfitting single batch")
        print("-----------------------------------------------------------------------")
        trainer.overfit_single_batch(train_loader)
#    else:
#        print("started the else part")
#        trainer.fit(train_loader, val_loader, epochs = epochs)
