import torch
import torch.nn as nn

from ignite.metrics import Loss, Accuracy

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from loguru import logger
from torchnlp.datasets import smt_dataset  # type: ignore

from slp.data.corpus import WordpieceCorpus
from slp.data.datasets import ClassificationCorpus
from slp.data.collators import SequenceClassificationCollator
from slp.data.transforms import ToTensor
from slp.modules.classifier import Classifier
from slp.modules.rnn import WordRNN
from slp.util.system import log_to_file
from slp.trainer import SequentialTrainer


EXPERIMENT_NAME = "smt-sentiment-classification"

log_to_file(f"logs/{EXPERIMENT_NAME}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

collate_fn = SequenceClassificationCollator(device="cpu")


if __name__ == "__main__":
    train, dev = smt_dataset(directory="../data/", train=True, dev=True)

    raw_train = [d["text"] for d in train]
    labels_train = [d["label"] for d in train]

    raw_dev = [d["text"] for d in dev]
    labels_dev = [d["label"] for d in dev]

    corpus_train = WordpieceCorpus(
        raw_train,
        bert_model="bert-base-uncased",
        lower=True,
    )

    corpus_dev = WordpieceCorpus(
        raw_dev,
        bert_model="bert-base-uncased",
        lower=True,
    )

    to_tensor = ToTensor(device="cpu")

    dataset_train = ClassificationCorpus(corpus_train, labels_train).map(to_tensor)
    dataset_dev = ClassificationCorpus(corpus_dev, labels_dev).map(to_tensor)

    train_loader = DataLoader(
        dataset_train,
        batch_size=8,
        num_workers=1,
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_fn,
    )

    dev_loader = DataLoader(
        dataset_dev,
        batch_size=8,
        num_workers=1,
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_fn,
    )

    encoder = WordRNN(
        256,
        vocab_size=corpus_train.vocab_size,
        embeddings_dim=300,
        bidirectional=True,
        merge_bi="sum",
        packed_sequence=True,
        attention=True,
    )

    model = Classifier(encoder, encoder.out_size, 3)

    optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    metrics = {"accuracy": Accuracy(), "loss": Loss(criterion)}
    trainer = SequentialTrainer(
        model,
        optimizer,
        checkpoint_dir="./checkpoints",
        experiment_name=EXPERIMENT_NAME,
        metrics=metrics,
        non_blocking=True,
        patience=5,
        loss_fn=criterion,
        device=DEVICE,
    )
    trainer.fit(train_loader, dev_loader, epochs=10)
