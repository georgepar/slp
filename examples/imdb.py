import torch
import torch.nn as nn

from ignite.metrics import Loss, Accuracy

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from torchnlp.datasets import imdb_dataset  # type: ignore

from slp import configure_logging
from slp.data.corpus import WordCorpus
from slp.data.datasets import CorpusDataset
from slp.data.collators import SequenceClassificationCollator
from slp.data.transforms import ToTensor
from slp.modules.classifier import Classifier
from slp.modules.rnn import WordRNN
from slp.trainer import SequentialTrainer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

collate_fn = SequenceClassificationCollator(device="cpu")

EXPERIMENT_NAME = "imdb-sentiment-classification"

configure_logging(f"logs/{EXPERIMENT_NAME}")


if __name__ == "__main__":
    train, dev = imdb_dataset(directory="../data/", train=True, test=True)

    raw_train = [d["text"] for d in train]
    labels_train = [d["sentiment"] for d in train]

    raw_dev = [d["text"] for d in dev]
    labels_dev = [d["sentiment"] for d in dev]

    corpus_train = WordCorpus(
        raw_train,
        limit_vocab_size=30000,
        embeddings_file="./cache/glove.6B.50d.txt",
        embeddings_dim=50,
        lower=True,
        lang="en_core_web_md",
    )

    # Train corpus vocabulary is forced on dev set
    corpus_dev = WordCorpus(
        raw_dev,
        limit_vocab_size=-1,
        embeddings=corpus_train.embeddings,
        word2idx=corpus_train.word2idx,
        idx2word=corpus_train.idx2word,
        lower=True,
        lang="en_core_web_md",
    )

    to_tensor = ToTensor(device="cpu")

    dataset_train = CorpusDataset(corpus_train, labels_train).map(to_tensor)
    dataset_dev = CorpusDataset(corpus_dev, labels_dev).map(to_tensor)

    train_loader = DataLoader(
        dataset_train,
        batch_size=64,
        num_workers=1,
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_fn,
    )

    dev_loader = DataLoader(
        dataset_dev,
        batch_size=64,
        num_workers=1,
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_fn,
    )

    encoder = WordRNN(
        256,
        embeddings=corpus_train.embeddings,
        bidirectional=True,
        merge_bi="sum",
        packed_sequence=True,
        finetune_embeddings=False,
        attention=True,
    )
    model = Classifier(encoder, encoder.out_size, 3)

    optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    metrics = {"accuracy": Accuracy(), "loss": Loss(criterion)}
    trainer = SequentialTrainer(
        model,
        optimizer,
        experiment_name=EXPERIMENT_NAME,
        checkpoint_dir="./checkpoints",
        metrics=metrics,
        non_blocking=True,
        patience=5,
        loss_fn=criterion,
        device=DEVICE,
    )
    trainer.fit(train_loader, dev_loader, epochs=10)
