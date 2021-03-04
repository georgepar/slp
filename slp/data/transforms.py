import os
import spacy
import torch

import sentencepiece as spm

from typing import List, Dict, Any, Optional

from transformers import AutoTokenizer
from spacy.attrs import ORTH

from slp.config.nlp import SPECIAL_TOKENS
from slp.util.pytorch import mktensor

# Avoid deadlocks for hugging face tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SentencepieceTokenizer(object):
    def __init__(
        self,
        lower: bool = True,
        model: Optional[Any] = None,
        prepend_bos: bool = False,
        append_eos: bool = False,
        specials: Optional[SPECIAL_TOKENS] = SPECIAL_TOKENS,  # type: ignore
    ):
        """Tokenize sentence using pretrained sentencepiece model

        Args:
            lower (bool): Lowercase string. Defaults to True.
            model (Optional[Any]): Sentencepiece model. Defaults to None.
            prepend_bos (bool): Prepend BOS for seq2seq. Defaults to False.
            append_eos (bool): Append EOS for seq2seq. Defaults to False.
            specials (Optional[SPECIAL_TOKENS]): Special tokens. Defaults to SPECIAL_TOKENS.
        """
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(model)
        self.specials = specials
        self.lower = lower
        self.vocab_size = self.tokenizer.get_piece_size()
        self.pre_id = []
        self.post_id = []
        if prepend_bos:
            self.pre_id.append(self.tokenizer.piece_to_id(self.specials.BOS.value))  # type: ignore
        if append_eos:
            self.post_id.append(self.tokenizer.piece_to_id(self.specials.EOS.value))  # type: ignore

    def __call__(self, x: str) -> List[int]:
        """Call to tokenize function

        Args:
            x (str): Input string

        Returns:
            List[int]: List of tokens ids
        """
        if self.lower:
            x = x.lower()
        ids: List[int] = self.pre_id + self.tokenizer.encode_as_ids(x) + self.post_id
        return ids


class HuggingFaceTokenizer(object):
    def __init__(
        self,
        lower: bool = True,
        model: str = "bert-base-uncased",
        add_special_tokens: bool = True,
    ):
        """Apply one of huggingface tokenizers to a string

        Args:
            lower (bool): Lowercase string. Defaults to True.
            model (str): Select transformer model. Defaults to "bert-base-uncased".
            add_special_tokens (bool): Insert special tokens to tokenized string. Defaults to True.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model, do_lower_case=lower)
        self.tokenizer.max_len = 65536  # hack to suppress warnings
        self.vocab_size = len(self.tokenizer.vocab)
        self.add_special_tokens = add_special_tokens

    def detokenize(self, x: List[int]) -> List[str]:
        """Convert list of token ids to list of tokens

        Args:
            x (List[int]): List of token ids

        Returns:
            List[str]: List of tokens
        """
        out: List[str] = self.tokenizer.convert_ids_to_tokens(x)
        return out

    def __call__(self, x: str) -> List[int]:
        """Call to tokenize function

        Args:
            x (str): Input string

        Returns:
            List[int]: List of token ids
        """
        out: List[int] = self.tokenizer.encode(
            x, add_special_tokens=self.add_special_tokens
        )
        return out


class SpacyTokenizer(object):
    def __init__(
        self,
        lower: bool = True,
        prepend_bos: bool = False,
        append_eos: bool = False,
        specials: Optional[SPECIAL_TOKENS] = SPECIAL_TOKENS,  # type: ignore
        lang: str = "en_core_web_sm",
    ):
        """Apply spacy tokenizer to str

        Args:
            lower (bool): Lowercase string. Defaults to True.
            prepend_bos (bool): Prepend BOS for seq2seq. Defaults to False.
            append_eos (bool): Append EOS for seq2seq. Defaults to False.
            specials (Optional[SPECIAL_TOKENS]): Special tokens. Defaults to SPECIAL_TOKENS.
            lang (str): Spacy language, e.g. el_core_web_sm, en_core_web_sm etc. Defaults to "en_core_web_md".
        """
        self.lower = lower
        self.specials = SPECIAL_TOKENS
        self.lang = lang
        self.pre_id = []
        self.post_id = []
        if prepend_bos:
            self.pre_id.append(self.specials.BOS.value)
        if append_eos:
            self.post_id.append(self.specials.EOS.value)
        self.nlp = self.get_nlp(name=lang, specials=specials)

    def get_nlp(
        self,
        name: str = "en_core_web_sm",
        specials: Optional[SPECIAL_TOKENS] = SPECIAL_TOKENS,  # type: ignore
    ) -> spacy.Language:
        """Get spacy nlp object for given lang and add SPECIAL_TOKENS

        Args:
            name (str): Spacy language, e.g. el_core_web_sm, en_core_web_sm etc. Defaults to "en_core_web_md".
            specials (Optional[SPECIAL_TOKENS]): Special tokens. Defaults to SPECIAL_TOKENS.

        Returns:
            spacy.Language: spacy text-processing pipeline
        """
        nlp = spacy.load(name)
        if specials is not None:
            for token in specials.to_list():
                nlp.tokenizer.add_special_case(token, [{ORTH: token}])
        return nlp

    def __call__(self, x: str) -> List[str]:
        """Call to tokenize function

        Args:
            x (str): Input string

        Returns:
            List[str]: List of tokens
        """
        if self.lower:
            x = x.lower()
        out: List[str] = (
            self.pre_id + [y.text for y in self.nlp.tokenizer(x)] + self.post_id
        )
        return out


class ToTokenIds(object):
    def __init__(
        self,
        word2idx: Dict[str, int],
        specials: Optional[SPECIAL_TOKENS] = SPECIAL_TOKENS,  # type: ignore
    ):
        """Convert List of tokens to list of token ids

        Args:
            word2idx (Dict[str, int]): Word to index mapping
            specials (Optional[SPECIAL_TOKENS]): Special tokens. Defaults to SPECIAL_TOKENS.
        """
        self.word2idx = word2idx
        self.unk_value = specials.UNK.value if specials is not None else "[UNK]"  # type: ignore

    def __call__(self, x: List[str]) -> List[int]:
        """Convert list of tokens to list of token ids

        Args:
            x (List[str]): List of tokens

        Returns:
            List[int]: List of token ids
        """
        return [
            self.word2idx[w] if w in self.word2idx else self.word2idx[self.unk_value]
            for w in x
        ]


class ReplaceUnknownToken(object):
    def __init__(
        self,
        old_unk: str = "<unk>",
        new_unk: str = SPECIAL_TOKENS.UNK.value,  # type: ignore
    ):
        """Replace existing unknown tokens in the vocab to [UNK]. Useful for wikitext

        Args:
            old_unk (str): Unk token in corpus. Defaults to "<unk>".
            new_unk (str): Desired unk value. Defaults to SPECIAL_TOKENS.UNK.value.
        """
        self.old_unk = old_unk
        self.new_unk = new_unk

    def __call__(self, x: List[str]) -> List[str]:
        """Convert <unk> in list of tokens to [UNK]

        Args:
            x (List[str]): List of tokens

        Returns:
            List[str]: List of tokens
        """
        return [w if w != self.old_unk else self.new_unk for w in x]


class ToTensor(object):
    def __init__(self, device: str = "cpu", dtype: torch.dtype = torch.long):
        """To tensor convertor

        Args:
            device (str): Device to map the tensor. Defaults to "cpu".
            dtype (torch.dtype): Type of resulting tensor. Defaults to torch.long.
        """
        self.device = device
        self.dtype = dtype

    def __call__(self, x: List[Any]) -> torch.Tensor:
        """Convert list of tokens or list of features to tensor

        Args:
            x (List[Any]): List of tokens or features

        Returns:
            torch.Tensor: Resulting tensor
        """
        return mktensor(x, device=self.device, dtype=self.dtype)
