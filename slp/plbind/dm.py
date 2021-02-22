import torch
import pytorch_lightning as pl

from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, DataLoader

from slp.data.corpus import WordCorpus, WordpieceCorpus
from slp.data.datasets import CorpusDataset
from slp.data.transforms import ToTensor


def split_data(dataset, test_size=0.2, seed=None):
    train, test = None, None

    if isinstance(dataset, torch.utils.data.Dataset):
        test_len = int(test_size * len(dataset))
        train_len = len(dataset) - test_len

        seed_generator = None
        if seed is not None:
            seed_generator = torch.Generator().manual_seed(seed)

        train, test = random_split(
            dataset, [train_len, test_len], generator=seed_generator
        )

    else:

        train, test = train_test_split(dataset, test_size=test_size, random_state=seed)

    return train, test


class PLDataModuleFromDatasets(pl.LightningDataModule):
    def __init__(
        self,
        train,
        val=None,
        test=None,
        val_percent=0.2,
        test_percent=0.2,
        batch_size=64,
        batch_size_eval=None,
        seed=None,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
        shuffle_eval=False,
        collate_fn=None,
    ):
        super(PLDataModuleFromDatasets, self).__init__()

        self.batch_size = batch_size

        if batch_size_eval is None:
            batch_size_eval = self.batch_size

        self.batch_size_eval = batch_size_eval

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.shuffle_eval = shuffle_eval
        self.collate_fn = collate_fn

        self.train = train
        self.val = val
        self.test = test

        if self.val is not None:
            logger.info(f"Using provided validation set")

        if self.test is not None:
            logger.info(f"Using provided test set")

        if self.test is None and val is None:
            testval_percent = test_percent + val_percent

            logger.info(
                f"No test or validation set provided. Creating random splits using {testval_percent * 100}% of training set with seed={seed}"
            )

            self.train, testval = split_data(
                self.train, test_size=testval_percent, seed=seed
            )

            test_percent = test_percent / testval_percent
            self.val, self.test = split_data(testval, test_size=test_percent, seed=seed)

        if self.val is None:
            logger.info(
                f"No validation set provided. Creating random split using {val_percent * 100}% of training set with seed={seed}"
            )
            self.train, self.val = split_data(
                self.train, test_size=val_percent, seed=seed
            )

        if self.test is None:
            logger.info(
                f"No test set provided. Creating random split using {test_percent * 100}% of training set with seed={seed}"
            )
            self.train, self.test = split_data(
                self.train, test_size=test_percent, seed=seed
            )

        logger.info(f"Using {len(self.train)} samples for training")
        logger.info(f"Using {len(self.val)} samples for validation")
        logger.info(f"Using {len(self.test)} samples for testing")

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size_eval,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=self.shuffle_eval,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size_eval,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=self.shuffle_eval,
            collate_fn=self.collate_fn,
        )


class PLDataModuleFromCorpus(PLDataModuleFromDatasets):
    def __init__(
        self,
        train,
        train_labels,
        val=None,
        val_labels=None,
        test=None,
        test_labels=None,
        val_percent=0.2,
        test_percent=0.2,
        batch_size=64,
        batch_size_eval=None,
        seed=None,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
        shuffle_eval=False,
        collate_fn=None,
        tokens="wordpieces",  # or "words"
        **corpus_args,
    ):
        train_data = list(zip(train, train_labels))
        val_data = None
        if val is not None:
            val_data = list(zip(val, val_labels))
        test_data = None
        if test is not None:
            test_data = list(zip(test, test_labels))

        super(PLDataModuleFromCorpus, self).__init__(
            train_data,
            val=val_data,
            test=test_data,
            val_percent=val_percent,
            test_percent=test_percent,
            batch_size=batch_size,
            batch_size_eval=batch_size_eval,
            seed=seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            shuffle_eval=shuffle_eval,
            collate_fn=collate_fn,
        )

        train_corpus, train_labels = zip(*self.train)
        val_corpus, val_labels = zip(*self.val)
        test_corpus, test_labels = zip(*self.test)

        accepted_token_types = ["words", "wordpieces"]
        corpus_cls = None
        if tokens == "words":
            logger.info('Selecting WordCorpus because tokens="words" was provided')
            corpus_cls = WordCorpus
        elif tokens == "wordpieces":
            logger.info(
                'Selecting WordpieceCorpus because tokens="wordpieces" was provided'
            )
            corpus_cls = WordpieceCorpus
        else:
            raise ValueError(
                f"tokens kwarg in {self.__class__.__name__} should be in {accepted_token_types}"
            )

        self.train_corpus = corpus_cls(train_corpus, **corpus_args)

        if tokens == "words":
            # Force train vocabulary on val & test
            corpus_args["embeddings"] = self.train_corpus.embeddings
            corpus_args["word2idx"] = self.train_corpus.word2idx
            corpus_args["idx2word"] = self.train_corpus.word2idx

            logger.info(
                "Forcing vocabulary from training set for validation and test sets."
            )

        self.val_corpus = corpus_cls(val_corpus, **corpus_args)
        self.test_corpus = corpus_cls(test_corpus, **corpus_args)

        to_tensor = ToTensor(device="cpu")

        self.train = CorpusDataset(self.train_corpus, train_labels).map(to_tensor)
        self.val = CorpusDataset(self.val_corpus, val_labels).map(to_tensor)
        self.test = CorpusDataset(self.test_corpus, test_labels).map(to_tensor)

    @property
    def embeddings(self):
        return self.train_corpus.embeddings

    @property
    def vocab_size(self):
        return self.train_corpus.vocab_size