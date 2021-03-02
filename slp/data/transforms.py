import os
import spacy
import torch

import sentencepiece as spm
from transformers import AutoTokenizer
from spacy.attrs import ORTH

from slp.config.nlp import SPECIAL_TOKENS
from slp.util.pytorch import mktensor

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SentencepieceTokenizer(object):
    def __init__(
        self,
        lower=True,
        model=None,
        prepend_bos=False,
        append_eos=False,
        specials=SPECIAL_TOKENS,
    ):
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(model)
        self.specials = specials
        self.lower = lower
        self.vocab_size = self.tokenizer.get_piece_size()
        self.pre_id = []
        self.post_id = []
        if prepend_bos:
            self.pre_id.append(self.tokenizer.piece_to_id(self.specials.BOS.value))
        if append_eos:
            self.post_id.append(self.tokenizer.piece_to_id(self.specials.EOS.value))

    def __call__(self, x):
        if self.lower:
            x = x.lower()
        ids = self.pre_id + self.tokenizer.encode_as_ids(x) + self.post_id
        return ids


class HuggingFaceTokenizer(object):
    def __init__(
        self,
        lower=True,
        model="bert-base-uncased",
        add_special_tokens=True,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model, do_lower_case=lower)
        self.tokenizer.max_len = 65536  # hack to suppress warnings
        self.vocab_size = len(self.tokenizer.vocab)
        self.add_special_tokens = add_special_tokens

    def detokenize(self, x):
        return self.tokenizer.convert_ids_to_tokens(x)

    def __call__(self, x):
        return self.tokenizer.encode(x, add_special_tokens=self.add_special_tokens)


class SpacyTokenizer(object):
    def __init__(
        self,
        lower=True,
        prepend_bos=False,
        append_eos=False,
        specials=SPECIAL_TOKENS,
        lang="en_core_web_sm",
    ):
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

    def get_nlp(self, name="en_core_web_sm", specials=SPECIAL_TOKENS):
        nlp = spacy.load(name)
        for control_token in map(lambda x: x.value, specials):
            nlp.tokenizer.add_special_case(control_token, [{ORTH: control_token}])
        return nlp

    def __call__(self, x):
        if self.lower:
            x = x.lower()
        x = self.pre_id + [y.text for y in self.nlp.tokenizer(x)] + self.post_id
        return x


class ToTokenIds(object):
    def __init__(self, word2idx, specials=SPECIAL_TOKENS):
        self.word2idx = word2idx
        self.specials = specials

    def __call__(self, x):
        return [
            self.word2idx[w]
            if w in self.word2idx
            else self.word2idx[self.specials.UNK.value]
            for w in x
        ]


class ReplaceUnknownToken(object):
    def __init__(self, old_unk="<unk>", new_unk=SPECIAL_TOKENS.UNK.value):
        self.old_unk = old_unk
        self.new_unk = new_unk

    def __call__(self, x):
        return [w if w != self.old_unk else self.new_unk for w in x]


class ToTensor(object):
    def __init__(self, device="cpu", dtype=torch.long):
        self.device = device
        self.dtype = dtype

    def __call__(self, x):
        return mktensor(x, device=self.device, dtype=self.dtype)
