import sentencepiece as spm
import spacy
import torch
from spacy.attrs import ORTH
from transformers import BertTokenizer

from slp.config import SPECIAL_TOKENS
from slp.util import mktensor


class CharacterTokenizer(object):
    def __init__(
        self,
        character_vocab,
        prepend_bos=False,
        append_eos=False,
        specials=SPECIAL_TOKENS,
    ):
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.vocab = character_vocab
        self.specials = specials
        self.vocab = [
            specials.PAD.value,
            specials.BOS.value,
            specials.EOS.value,
        ] + self.vocab
        self.c2i = {c: i for i, c in enumerate(self.vocab)}
        self.i2c = {i: c for i, c in enumerate(self.vocab)}

    def detokenize(self, ids):
        return "".join([self.i2c[i] for i in ids])

    def __call__(self, sentence):
        chars = []
        prev_char = ""

        for c in sentence:
            if c not in self.vocab or (prev_char == " " and c == " "):
                continue
            chars.append(self.c2i[c])

        if self.prepend_bos:
            chars = [self.c2i[self.specials.BOS.value]] + chars

        if self.append_eos:
            chars = chars + [self.c2i[self.specials.EOS.value]]

        return chars


class SentencepieceTokenizer(object):
    def __init__(
        self,
        lower=True,
        model=None,
        prepend_bos=False,
        prepend_cls=False,
        append_eos=False,
        specials=SPECIAL_TOKENS,
    ):
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(model)
        self.specials = specials
        self.lower = lower
        self.vocab_size = self.tokenizer.get_piece_size()

        if prepend_cls and prepend_bos:
            raise ValueError("prepend_bos and prepend_cls are" " mutually exclusive")
        self.pre_id = []
        self.post_id = []

        if prepend_cls:
            self.pre_id.append(self.tokenizer.piece_to_id(self.specials.CLS.value))

        if prepend_bos:
            self.pre_id.append(self.tokenizer.piece_to_id(self.specials.BOS.value))

        if append_eos:
            self.post_id.append(self.tokenizer.piece_to_id(self.specials.EOS.value))

    def detokenize(self, x):
        return self.tokenizer.decode(x)

    def __call__(self, x):
        if self.lower:
            x = x.lower()
        ids = self.pre_id + self.tokenizer.encode_as_ids(x) + self.post_id

        return ids


class WordpieceTokenizer(object):
    def __init__(
        self,
        lower=True,
        bert_model="bert-base-uncased",
        prepend_cls=False,
        prepend_bos=False,
        append_eos=False,
        specials=SPECIAL_TOKENS,
    ):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=lower)
        # self.tokenizer.max_len = 1024  # hack to suppress warnings
        self.vocab = [k for k in self.tokenizer.vocab.keys()]
        self.c2i = {c: i for c, i in enumerate(self.tokenizer.vocab.items())}
        self.i2c = {i: c for c, i in enumerate(self.tokenizer.vocab.items())}

        self.specials = specials
        self.vocab_size = len(self.tokenizer.vocab)
        self.pre_id = []
        self.post_id = []

        if prepend_cls and prepend_bos:
            raise ValueError("prepend_bos and prepend_cls are" " mutually exclusive")

        if prepend_cls:
            self.pre_id.append(self.specials.CLS.value)

        if prepend_bos:
            self.pre_id.append(self.specials.BOS.value)

        if append_eos:
            self.post_id.append(self.specials.EOS.value)

    def detokenize(self, x):
        return self.tokenizer.convert_ids_to_tokens(x)

    def __call__(self, x):
        x = self.pre_id + self.tokenizer.tokenize(x) + self.post_id

        return self.tokenizer.convert_tokens_to_ids(x)


class SpacyTokenizer(object):
    def __init__(
        self,
        lower=True,
        prepend_cls=False,
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

        if prepend_cls and prepend_bos:
            raise ValueError("prepend_bos and prepend_cls are" " mutually exclusive")

        if prepend_cls:
            self.pre_id.append(self.specials.CLS.value)

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
