import random

import torch
import torch.nn as nn
from ignite.metrics import Accuracy, Loss
from sklearn.preprocessing import LabelEncoder
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchnlp.datasets import imdb_dataset  # type: ignore

from slp.data.collators import TransformerSequenceClassificationCollator
from slp.data.transforms import SpacyTokenizer, ToTensor, ToTokenIds
from slp.modules.classifier import AttentionClassifier as Classifier
from slp.modules.transformer import TransformerEncoder
from slp.trainer import SequentialTrainer
from slp.util import log
from slp.util.embeddings import EmbeddingsLoader

DEBUG = False
MAX_LENGTH = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64


class DatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transforms = []
        self.label_encoder = (LabelEncoder()
                              .fit([d['sentiment'] for d in dataset]))

    def map(self, t):
        self.transforms.append(t)

        return self

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        datum = self.dataset[idx]
        text, target = datum['text'], datum['sentiment']
        target = self.label_encoder.transform([target])[0]

        for t in self.transforms:
            text = t(text)

        return text, target


collate_fn = TransformerSequenceClassificationCollator(device='cpu', max_length=MAX_LENGTH)


if __name__ == '__main__':

    loader = EmbeddingsLoader(
       './cache/glove.840B.300d.txt', 300, max_vectors=500000)
    word2idx, _, embeddings = loader.load()

    embeddings = torch.from_numpy(embeddings)

    tokenizer = SpacyTokenizer()
    to_token_ids = ToTokenIds(word2idx)
    to_tensor = ToTensor(device='cpu')

    def create_dataloader(d):
        d = (DatasetWrapper(d).map(tokenizer).map(to_token_ids).map(to_tensor))

        return DataLoader(
            d, batch_size=32,
            num_workers=1,
            pin_memory=True,
            shuffle=True,
            collate_fn=collate_fn)

    train_loader, test_loader = map(
        create_dataloader,
        imdb_dataset(directory='./data/', train=True, test=True))


    model = Classifier(
        TransformerEncoder(
            embeddings=embeddings,
            finetune_embeddings=True,
            max_length=512,
            num_layers=4,
            hidden_size=128,
            num_heads=8,
            inner_size=256,
            dropout=.1,
            device=DEVICE
        ),
        128, 3
    )

    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4
    )
    criterion = nn.CrossEntropyLoss()
    metrics = {
        'accuracy': Accuracy(),
        'loss': Loss(criterion)
    }

    trainer = SequentialTrainer(
        model,
        optimizer,
        checkpoint_dir=None, # './checkpoints' if not DEBUG else None,
        metrics=metrics,
        non_blocking=True,
        retain_graph=True,
        patience=5,
        loss_fn=criterion,
        device=DEVICE,
        extra_args={"max_length": MAX_LENGTH}
    )

    if DEBUG:
        log.info('Starting end to end test')
        print('--------------------------------------------------------------')
        trainer.fit_debug(train_loader, dev_loader)
        log.info('Overfitting single batch')
        print('--------------------------------------------------------------')
        trainer.overfit_single_batch(train_loader)
    else:
        trainer.fit(train_loader, test_loader, epochs=10)
