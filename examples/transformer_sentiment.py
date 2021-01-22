import random

import torch
import torch.nn as nn
from ignite.metrics import Accuracy, Loss
from sklearn.preprocessing import LabelEncoder
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchnlp.datasets import imdb_dataset, smt_dataset  # type: ignore

from slp.data.collators import TransformerSequenceClassificationCollator
from slp.data.transforms import SpacyTokenizer, ToTensor, ToTokenIds
from slp.modules.classifier import AttentionClassifier as Classifier
from slp.modules.transformer import TransformerEncoder
from slp.trainer import SequentialTrainer
from slp.util import log
from slp.util.embeddings import EmbeddingsLoader

LIMIT_TRAIN_DATASET = -1  # Test with different number of samples. -1 Takes the whole dataset


DEBUG = False
MAX_LENGTH = 500
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-4

MAX_EMBEDDING_VECTORS = 500000

FINETUNE_EMBEDDINGS = True  # Leave true else not training
NUM_LAYERS = 4
HIDDEN_SIZE = 256
NUM_HEADS = 8
INNER_SIZE = 2 * HIDDEN_SIZE
DROPOUT = .1

DATASET = "smt"  # "imdb" or "smt"


class DatasetWrapper(Dataset):
    def __init__(self, data, dataset="imdb"):
        self.dataset = data
        self.transforms = []
        self.label_key = "sentiment" if dataset == "imdb" else "label"
        self.label_encoder = (LabelEncoder()
                              .fit([d[self.label_key] for d in data]))

    def map(self, t):
        self.transforms.append(t)

        return self

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        datum = self.dataset[idx]
        text, target = datum['text'], datum[self.label_key]
        target = self.label_encoder.transform([target])[0]

        for t in self.transforms:
            text = t(text)

        return text, target


collate_fn = TransformerSequenceClassificationCollator(device='cpu', max_length=MAX_LENGTH)


if __name__ == '__main__':

    loader = EmbeddingsLoader(
       './cache/glove.840B.300d.txt', 300, max_vectors=MAX_EMBEDDING_VECTORS)
    word2idx, _, embeddings = loader.load()

    embeddings = torch.from_numpy(embeddings)

    tokenizer = SpacyTokenizer()
    to_token_ids = ToTokenIds(word2idx)
    to_tensor = ToTensor(device='cpu')

    def create_dataloader(d, shuffle=False):
        d = (DatasetWrapper(d, dataset=DATASET).map(tokenizer).map(to_token_ids).map(to_tensor))

        return DataLoader(
            d, batch_size=32,
            num_workers=1,
            pin_memory=True,
            shuffle=shuffle,
            drop_last=False,
            collate_fn=collate_fn)

    if DATASET == "imdb":
        traindev, test = imdb_dataset(directory='./data/', train=True, test=True)
        num_classes = 3

        import random

        def train_test_split(data, test_size=.2):
            random.shuffle(data)
            train_size = int((1 - test_size) * len(data))
            train = data[:train_size]
            test = data[train_size:]

            return train, test

        train, dev = train_test_split(traindev)
        train = train[:LIMIT_TRAIN_DATASET]

    else:
        train, dev, test = smt_dataset(
            directory='./data/', train=True, dev=True, test=True,
            fine_grained=False
        )

        train = train[:LIMIT_TRAIN_DATASET]
        num_classes = 3


    train_loader = create_dataloader(train, shuffle=True)
    dev_loader = create_dataloader(dev, shuffle=False)
    test_loader = create_dataloader(test, shuffle=False)

    model = Classifier(
        TransformerEncoder(
            embeddings=embeddings,
            finetune_embeddings=FINETUNE_EMBEDDINGS,
            max_length=MAX_LENGTH + 20,  # Always a good idea to add a safety here.
            num_layers=NUM_LAYERS,
            hidden_size=HIDDEN_SIZE,
            num_heads=NUM_HEADS,
            inner_size=INNER_SIZE,
            dropout=DROPOUT,
            device=DEVICE
        ),
        HIDDEN_SIZE, num_classes
    )

    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE
    )
    criterion = nn.CrossEntropyLoss()
    metrics = {
        'accuracy': Accuracy(),
        'loss': Loss(criterion)
    }

    trainer = SequentialTrainer(
        model,
        optimizer,
        checkpoint_dir='./checkpoints' if not DEBUG else None,
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
        trainer.fit(train_loader, dev_loader, epochs=EPOCHS)


        try:
            del trainer
        except Exception as e:
            pass


        trainer = SequentialTrainer(
            model,
            optimizer,
            checkpoint_dir='./checkpoints',
            metrics=metrics,
            loss_fn=criterion,
            device=DEVICE,
            extra_args={"max_length": MAX_LENGTH}
        )

        predictions, targets = trainer.predict(test_loader)

        predictions = predictions.argmax(-1).detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        from sklearn.metrics import accuracy_score
        print("Test Accuracy:")
        print(accuracy_score(predictions, targets))
