import argparse

import torch
import pytorch_lightning as pl

from transformers import ALL_PRETRAINED_CONFIG_ARCHIVE_MAP

from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, DataLoader

from slp.data.corpus import WordCorpus, WordpieceCorpus, TokenizedCorpus
from slp.data.datasets import CorpusDataset, CorpusLMDataset
from slp.data.transforms import ToTensor
from slp.util.types import dir_path


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
        batch_size=1,
        batch_size_eval=None,
        seed=None,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
        sampler_train=None,
        sampler_val=None,
        sampler_test=None,
        batch_sampler_train=None,
        batch_sampler_val=None,
        batch_sampler_test=None,
        shuffle_eval=False,
        collate_fn=None,
    ):
        super(PLDataModuleFromDatasets, self).__init__()

        if batch_sampler_train is not None and sampler_train is not None:
            raise ValueError(
                "You provided both a sampler and a batch sampler for the train set. These are mutually exclusive"
            )

        if batch_sampler_val is not None and sampler_val is not None:
            raise ValueError(
                "You provided both a sampler and a batch sampler for the validation set. These are mutually exclusive"
            )
        if batch_sampler_test is not None and sampler_test is not None:
            raise ValueError(
                "You provided both a sampler and a batch sampler for the test set. These are mutually exclusive"
            )
        self.sampler_train = sampler_train
        self.sampler_val = sampler_val
        self.sampler_test = sampler_test
        self.batch_sampler_train = batch_sampler_train
        self.batch_sampler_val = batch_sampler_val
        self.batch_sampler_test = batch_sampler_test
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        self.shuffle_eval = shuffle_eval
        self.collate_fn = collate_fn

        self.batch_size = batch_size

        if batch_size_eval is None:
            batch_size_eval = self.batch_size

        self.batch_size_eval = batch_size_eval

        self.train = train
        self.val = val
        self.test = test

        if self.val is not None:
            logger.info(f"Using provided validation set")

        if self.test is not None:
            logger.info(f"Using provided test set")

        if self.test is None and val is None:
            assert (
                val_percent is not None and val_percent > 0
            ), "You should either provide a validation set or a val set percentage"

            assert (
                test_percent is not None and test_percent > 0
            ), "You should either provide a test set or a test set percentage"

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
            assert (
                val_percent is not None and val_percent > 0
            ), "You should either provide a validation set or a val set percentage"

            logger.info(
                f"No validation set provided. Creating random split using {val_percent * 100}% of training set with seed={seed}"
            )
            self.train, self.val = split_data(
                self.train, test_size=val_percent, seed=seed
            )

        if self.test is None:
            assert (
                test_percent is not None and test_percent > 0
            ), "You should either provide a test set or a test set percentage"

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
            batch_size=self.batch_size if self.batch_sampler_train is None else 1,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last and (self.batch_sampler_train is None),
            sampler=self.sampler_train,
            batch_sampler=self.batch_sampler_train,
            shuffle=(self.batch_sampler_train is None) and (self.sampler_train is None),
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        val = DataLoader(
            self.val,
            batch_size=self.batch_size_eval if self.batch_sampler_val is None else 1,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last and (self.batch_sampler_val is None),
            sampler=self.sampler_val,
            batch_sampler=self.batch_sampler_val,
            shuffle=(
                self.shuffle_eval
                and (self.batch_sampler_val is None)
                and (self.sampler_val is None)
            ),
            collate_fn=self.collate_fn,
        )

        return val

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size_eval if self.batch_sampler_test is None else 1,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last and (self.batch_sampler_test is None),
            sampler=self.sampler_test,
            batch_sampler=self.batch_sampler_test,
            shuffle=(
                self.shuffle_eval
                and (self.batch_sampler_test is None)
                and (self.sampler_test is None)
            ),
            collate_fn=self.collate_fn,
        )

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--val-percent",
            dest="data.val_percent",
            type=float,
            help="Percent of validation data to be randomly split from the training set, if no validation set is provided",
        )

        parser.add_argument(
            "--test-percent",
            dest="data.test_percent",
            type=float,
            help="Percent of test data to be randomly split from the training set, if no test set is provided",
        )

        parser.add_argument(
            "--bsz", dest="data.batch_size", type=int, help="Training batch size"
        )

        parser.add_argument(
            "--bsz-eval",
            dest="data.batch_size_eval",
            type=int,
            help="Evaluation batch size",
        )

        parser.add_argument(
            "--num-workers",
            dest="data.num_workers",
            type=int,
            default=1,
            help="Number of workers to be used in the DataLoader",
        )

        parser.add_argument(
            "--pin-memory",
            dest="data.pin_memory",
            action="store_true",
            help="Pin data to GPU memory for faster data loading",
        )

        parser.add_argument(
            "--drop-last",
            dest="data.drop_last",
            action="store_true",
            help="Drop last incomplete batch",
        )

        parser.add_argument(
            "--shuffle-eval",
            dest="data.shuffle_eval",
            action="store_true",
            help="Shuffle val & test sets",
        )
        return parser


class PLDataModuleFromCorpus(PLDataModuleFromDatasets):
    accepted_tokenizers = ["tokenized", "spacy"] + list(
        ALL_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()
    )

    def __init__(
        self,
        train,
        train_labels=None,
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
        sampler_train=None,
        sampler_val=None,
        sampler_test=None,
        batch_sampler_train=None,
        batch_sampler_val=None,
        batch_sampler_test=None,
        language_model=False,
        collate_fn=None,
        tokenizer="spacy",
        **corpus_args,
    ):
        self.language_model = language_model
        if not language_model and train_labels is None:
            raise ValueError(
                "You should provide train labels if not performing language modeling"
            )

        if language_model:
            train_labels = train[1:]
            train = train[0:-1]
            if val is not None:
                val_labels = val[1:]
                val = val[0:-1]
            if test is not None:
                test_labels = test[1:]
                test = test[0:-1]

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
            sampler_train=sampler_train,
            sampler_val=sampler_val,
            sampler_test=sampler_test,
            batch_sampler_train=batch_sampler_train,
            batch_sampler_val=batch_sampler_val,
            batch_sampler_test=batch_sampler_test,
            collate_fn=collate_fn,
        )

        train_corpus, train_labels = zip(*self.train)
        val_corpus, val_labels = zip(*self.val)
        test_corpus, test_labels = zip(*self.test)

        corpus_cls = None
        if tokenizer not in self.accepted_tokenizers:
            raise ValueError(
                f"tokenizer kwarg in {self.__class__.__name__} should be one of {accepted_tokenizers}"
            )

        if tokenizer == "spacy":
            logger.info('Selecting WordCorpus because tokenizer="spacy" was provided')
            corpus_cls = WordCorpus
        elif tokenizer == "tokenized":
            logger.info(
                'Selecting TokenizedCorpus because tokenizer="tokenized" was provided'
            )
            corpus_cls = TokenizedCorpus
        else:
            logger.info(
                "Selecting WordpieceCorpus because a huggingface tokenizer was provided"
            )
            corpus_cls = WordpieceCorpus
            corpus_args["tokenizer_model"] = tokenizer

        self.train_corpus = corpus_cls(train_corpus, **corpus_args)

        if tokenizer == "spacy" or tokenizer == "tokenized":
            # Force train vocabulary on val & test
            corpus_args["word2idx"] = self.train_corpus.word2idx

            if tokenizer == "spacy":
                corpus_args["embeddings"] = self.train_corpus.embeddings
                corpus_args["idx2word"] = self.train_corpus.word2idx

            logger.info(
                "Forcing vocabulary from training set for validation and test sets."
            )

        self.val_corpus = corpus_cls(val_corpus, **corpus_args)
        self.test_corpus = corpus_cls(test_corpus, **corpus_args)

        to_tensor = ToTensor(device="cpu")

        if self.language_model:
            self.train = CorpusLMDataset(self.train_corpus).map(to_tensor)
            self.val = CorpusLMDataset(self.val_corpus).map(to_tensor)
            self.test = CorpusLMDataset(self.test_corpus).map(to_tensor)
        else:
            self.train = CorpusDataset(self.train_corpus, train_labels).map(to_tensor)
            self.val = CorpusDataset(self.val_corpus, val_labels).map(to_tensor)
            self.test = CorpusDataset(self.test_corpus, test_labels).map(to_tensor)

    @property
    def embeddings(self):
        return self.train_corpus.embeddings

    @property
    def vocab_size(self):
        return self.train_corpus.vocab_size

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = super(PLDataModuleFromCorpus, cls).add_argparse_args(parent_parser)
        parser.add_argument(
            "--tokenizer",
            dest="data.tokenizer",
            type=str.lower,
            # Corpus can already be tokenized, you can use spacy for word tokenization or any tokenizer from hugging face
            choices=cls.accepted_tokenizers,
            help="Token type. The tokenization will happen at this level.",
        )

        # Only when tokenizer == spacy
        parser.add_argument(
            "--limit-vocab",
            dest="data.limit_vocab_size",
            type=int,
            default=-1,
            help="Limit vocab size. -1 means use the whole vocab. Applicable only when --tokenizer=spacy",
        )

        parser.add_argument(
            "--embeddings-file",
            dest="data.embeddings_file",
            type=dir_path,
            help="Path to file with pretrained embeddings. Applicable only when --tokenizer=spacy",
        )

        parser.add_argument(
            "--embeddings-dim",
            dest="data.embeddings_dim",
            type=int,
            help="Embedding dim of pretrained embeddings. Applicable only when --tokenizer=spacy",
        )

        parser.add_argument(
            "--lang",
            dest="data.lang",
            type=str,
            default="en_core_web_md",
            help="Language for spacy tokenizer, e.g. en_core_web_md. Applicable only when --tokenizer=spacy",
        )

        parser.add_argument(
            "--add-specials",
            dest="data.add_special_tokens",
            action="store_true",
            help="Add special tokens for hugging face tokenizers",
        )

        # Generic args
        parser.add_argument(
            "--lower",
            dest="data.lower",
            action="store_true",
            help="Convert to lowercase.",
        )

        parser.add_argument(
            "--prepend-bos",
            dest="data.prepend_bos",
            action="store_true",
            help="Prepend [BOS] token",
        )

        parser.add_argument(
            "--append-eos",
            dest="data.append_eos",
            action="store_true",
            help="Append [EOS] token",
        )

        parser.add_argument(
            "--max-sentence-length",
            dest="data.max_len",
            type=int,
            default=-1,
            help="Maximum allowed sentence length. -1 means use the whole sentence",
        )

        return parser
