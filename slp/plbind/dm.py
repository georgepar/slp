import argparse

import numpy as np
import torch
import pytorch_lightning as pl

from transformers import ALL_PRETRAINED_CONFIG_ARCHIVE_MAP

from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, DataLoader, Dataset, Sampler, BatchSampler

from collections.abc import Sized

from typing import Union, List, Dict, Optional, Any, Tuple, Callable, Iterable

from slp.data.corpus import WordCorpus, HfCorpus, TokenizedCorpus
from slp.data.datasets import CorpusDataset, CorpusLMDataset
from slp.data.transforms import ToTensor
from slp.util.types import dir_path

DatasetType = Union[Dataset, List[Any]]


def split_data(dataset, test_size, seed):
    """Train-test split of dataset.

    Dataset can be either a torch.utils.data.Dataset or a list

    Args:
        dataset (Union[Dataset, List]): Input dataset
        test_size (float): Size of the test set. Defaults to 0.2.
        seed (int): Optional seed for deterministic run. Defaults to None.

    Returns:
        Tuple[Union[Dataset, List], Union[Dataset, List]: (train set, test set)
    """
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
        train: Dataset,
        val: Dataset = None,
        test: Dataset = None,
        val_percent: float = 0.2,
        test_percent: float = 0.2,
        batch_size: int = 1,
        batch_size_eval: Optional[int] = None,
        seed: Optional[int] = None,
        num_workers: int = 1,
        pin_memory: bool = True,
        drop_last: bool = False,
        sampler_train: Sampler = None,
        sampler_val: Sampler = None,
        sampler_test: Sampler = None,
        batch_sampler_train: BatchSampler = None,
        batch_sampler_val: BatchSampler = None,
        batch_sampler_test: BatchSampler = None,
        shuffle_eval: bool = False,
        collate_fn: Optional[Callable[..., Any]] = None,
    ):
        """LightningDataModule wrapper for generic torch.utils.data.Dataset

        If val or test Datasets are not provided, this class will split
        val_pecent and test_percent of the train set respectively to create them

        Args:
            train (Dataset): Train set
            val (Dataset): Validation set. Defaults to None.
            test (Dataset): Test set. Defaults to None.
            val_percent (float): Percent of train to be used for validation if no validation set is given. Defaults to 0.2.
            test_percent (float): Percent of train to be used for test set if no test set is given. Defaults to 0.2.
            batch_size (int): Training batch size. Defaults to 1.
            batch_size_eval (Optional[int]): Validation and test batch size. Defaults to None.
            seed (Optional[int]): Seed for deterministic run. Defaults to None.
            num_workers (int): Number of workers in the DataLoader. Defaults to 1.
            pin_memory (bool): Pin tensors to GPU memory. Defaults to True.
            drop_last (bool): Drop last incomplete batch. Defaults to False.
            sampler_train (Sampler): Sampler for train loader. Defaults to None.
            sampler_val (Sampler): Sampler for validation loader. Defaults to None.
            sampler_test (Sampler): Sampler for test loader. Defaults to None.
            batch_sampler_train (BatchSampler): Batch sampler for train loader. Defaults to None.
            batch_sampler_val (BatchSampler): Batch sampler for validation loader. Defaults to None.
            batch_sampler_test (BatchSampler): Batch sampler for test loader. Defaults to None.
            shuffle_eval (bool): Shuffle validation and test dataloaders. Defaults to False.
            collate_fn (Callable[..., Any]): Collator function. Defaults to None.

        Raises:
            ValueError: If both mutually exclusive sampler_train and batch_sampler_train are provided
            ValueError: If both mutually exclusive sampler_val and batch_sampler_val are provided
            ValueError: If both mutually exclusive sampler_test and batch_sampler_test are provided
        """
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

        logger.info(f"Using {len(self.train)} samples for training")  # type: ignore
        logger.info(f"Using {len(self.val)} samples for validation")  # type: ignore
        logger.info(f"Using {len(self.test)} samples for testing")  # type: ignore

    def train_dataloader(self) -> DataLoader:
        """Configure train DataLoader

        Returns:
            DataLoader: Pytorch DataLoader for train set
        """
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
        """Configure validation DataLoader

        Returns:
            DataLoader: Pytorch DataLoader for validation set
        """
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
        """Configure test DataLoader

        Returns:
            DataLoader: Pytorch DataLoader for test set
        """
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
    def add_argparse_args(
        cls, parent_parser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        """Augment input parser with arguments for data loading

        Args:
            parent_parser (argparse.ArgumentParser): Parser created by the user

        Returns:
            argparse.ArgumentParser: Augmented parser
        """
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
    accepted_tokenizers: List[str] = ["tokenized", "spacy"] + list(
        ALL_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()
    )

    def __init__(
        self,
        train: List,
        train_labels: Optional[List] = None,
        val: Optional[List] = None,
        val_labels: Optional[List] = None,
        test: Optional[List] = None,
        test_labels: Optional[List] = None,
        val_percent: float = 0.2,
        test_percent: float = 0.2,
        batch_size: int = 64,
        batch_size_eval: int = None,
        seed: int = None,
        num_workers: int = 1,
        pin_memory: bool = True,
        drop_last: bool = False,
        shuffle_eval: bool = False,
        sampler_train: Sampler = None,
        sampler_val: Sampler = None,
        sampler_test: Sampler = None,
        batch_sampler_train: BatchSampler = None,
        batch_sampler_val: BatchSampler = None,
        batch_sampler_test: BatchSampler = None,
        collate_fn: Optional[Callable[..., Any]] = None,
        language_model: bool = False,
        tokenizer: str = "spacy",
        **corpus_args,
    ):
        """Wrap raw corpus in a LightningDataModule

        * This handles the selection of the appropriate corpus class based on the tokenizer argument.
        * If language_model=True it uses the appropriate dataset from slp.data.datasets.
        * Uses the PLDataModuleFromDatasets to split the val and test sets if not provided

        Args:
            train (List): Raw train corpus
            train_labels (Optional[List]): Train labels. Defaults to None.
            val (Optional[List]): Raw validation corpus. Defaults to None.
            val_labels (Optional[List]): Validation labels. Defaults to None.
            test (Optional[List]): Raw test corpus. Defaults to None.
            test_labels (Optional[List]): Test labels. Defaults to None.
            val_percent (float): Percent of train to be used for validation if no validation set is given. Defaults to 0.2.
            test_percent (float): Percent of train to be used for test set if no test set is given. Defaults to 0.2.
            batch_size (int): Training batch size. Defaults to 1.
            batch_size_eval (Optional[int]): Validation and test batch size. Defaults to None.
            seed (Optional[int]): Seed for deterministic run. Defaults to None.
            num_workers (int): Number of workers in the DataLoader. Defaults to 1.
            pin_memory (bool): Pin tensors to GPU memory. Defaults to True.
            drop_last (bool): Drop last incomplete batch. Defaults to False.
            sampler_train (Sampler): Sampler for train loader. Defaults to None.
            sampler_val (Sampler): Sampler for validation loader. Defaults to None.
            sampler_test (Sampler): Sampler for test loader. Defaults to None.
            batch_sampler_train (BatchSampler): Batch sampler for train loader. Defaults to None.
            batch_sampler_val (BatchSampler): Batch sampler for validation loader. Defaults to None.
            batch_sampler_test (BatchSampler): Batch sampler for test loader. Defaults to None.
            shuffle_eval (bool): Shuffle validation and test dataloaders. Defaults to False.
            collate_fn (Callable[..., Any]): Collator function. Defaults to None.
            language_model (bool): Use corpus for Language Modeling. Defaults to False.
            tokenizer (str): Select one of the cls.accepted_tokenizers. Defaults to "spacy".

        Raises:
            ValueError: [description]
            ValueError: [description]
        """
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

        train_data = (
            list(zip(train, train_labels)) if train_labels is not None else train
        )
        val_data = None
        if val is not None:
            val_data = list(zip(val, val_labels)) if val_labels is not None else val
        test_data = None
        if test is not None:
            test_data = (
                list(zip(test, test_labels)) if test_labels is not None else test
            )

        super(PLDataModuleFromCorpus, self).__init__(
            train_data,  # type: ignore
            val=val_data,  # type: ignore
            test=test_data,  # type: ignore
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

        train_corpus, train_labels = zip(*self.train)  # type: ignore
        val_corpus, val_labels = zip(*self.val)  # type: ignore
        test_corpus, test_labels = zip(*self.test)  # type: ignore

        if tokenizer not in self.accepted_tokenizers:
            raise ValueError(
                f"tokenizer kwarg in {self.__class__.__name__} should be one of {self.accepted_tokenizers}"
            )

        if tokenizer == "spacy":
            logger.info('Selecting WordCorpus because tokenizer="spacy" was provided')
            corpus_cls = WordCorpus  # type: ignore
        elif tokenizer == "tokenized":
            logger.info(
                'Selecting TokenizedCorpus because tokenizer="tokenized" was provided'
            )
            corpus_cls = TokenizedCorpus  # type: ignore
        else:
            logger.info(
                "Selecting HfCorpus because a huggingface tokenizer was provided"
            )
            corpus_cls = HfCorpus  # type: ignore
            corpus_args["tokenizer_model"] = tokenizer

        self.train_corpus = corpus_cls(train_corpus, **corpus_args)  # type: ignore

        if tokenizer == "spacy" or tokenizer == "tokenized":
            # Force train vocabulary on val & test
            corpus_args["word2idx"] = self.train_corpus.word2idx

            if tokenizer == "spacy":
                corpus_args["embeddings"] = self.train_corpus.embeddings
                corpus_args["idx2word"] = self.train_corpus.word2idx

            logger.info(
                "Forcing vocabulary from training set for validation and test sets."
            )

        self.val_corpus = corpus_cls(val_corpus, **corpus_args)  # type: ignore
        self.test_corpus = corpus_cls(test_corpus, **corpus_args)  # type: ignore

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
    def embeddings(self) -> Optional[np.ndarray]:
        """Embeddings matrix

        Returns:
            Optional[np.ndarray]: Embeddings matrix
        """
        emb: Optional[np.ndarray] = self.train_corpus.embeddings
        return emb

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary

        Returns:
            int: Number of tokens in the vocabulary
        """
        vsz: int = self.train_corpus.vocab_size
        return vsz

    @classmethod
    def add_argparse_args(cls, parent_parser):
        """Augment input parser with arguments for data loading and corpus processing

        Args:
            parent_parser (argparse.ArgumentParser): Parser created by the user

        Returns:
            argparse.ArgumentParser: Augmented parser
        """
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
