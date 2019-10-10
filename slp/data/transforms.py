import spacy
import torch

from pytorch_pretrained_bert import BertTokenizer
from spacy.attrs import ORTH

from slp.config import SPECIAL_TOKENS
from slp.util import mktensor


class WordpieceTokenizer(object):
    def __init__(self, lower=True, bert_model='bert-base-uncased',
                 specials=SPECIAL_TOKENS):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model,
                                                       do_lower_case=lower)
        self.tokenizer.max_len = 1024  # hack to suppress warnings
        self.specials = specials

    def __call__(self, x):
        x = [self.specials.CLS.value] + self.tokenizer.tokenize(x)
        return self.tokenizer.convert_tokens_to_ids(x)


class SpacyTokenizer(object):
    def __init__(self,
                 lower=True,
                 specials=SPECIAL_TOKENS,
                 lang='en_core_web_sm'):
        self.lower = lower
        self.specials = SPECIAL_TOKENS
        self.lang = lang
        self.nlp = self.get_nlp(name=lang, specials=specials)

    def get_nlp(self, name="en_core_web_sm", specials=SPECIAL_TOKENS):
        nlp = spacy.load(name)
        for control_token in map(lambda x: x.value, specials):
            nlp.tokenizer.add_special_case(
                control_token, [{ORTH: control_token}])
        return nlp

    def __call__(self, x):
        if self.lower:
            x = x.lower()
        x = [y.text for y in self.nlp.tokenizer(x)]
        if self.specials.has_token('BOS'):
            x = [self.specials.BOS.value] + x
        if self.specials.has_token('EOS'):
            x = x + [self.specials.EOS.value]
        return x


class ToTokenIds(object):
    def __init__(self, word2idx, specials=SPECIAL_TOKENS):
        self.word2idx = word2idx
        self.specials = specials

    def __call__(self, x):
        return [self.word2idx[w]
                if w in self.word2idx
                else self.word2idx[self.specials.UNK.value]
                for w in x]


class ReplaceUnknownToken(object):
    def __init__(self, old_unk='<unk>', new_unk=SPECIAL_TOKENS.UNK.value):
        self.old_unk = old_unk
        self.new_unk = new_unk

    def __call__(self, x):
        return [w if w != self.old_unk else self.new_unk for w in x]


class ReplaceUnknownsTransform(object):
    def __init__(self, vocabulary, unk=SPECIAL_TOKENS.UNK.value):
        self.vocabulary = vocabulary
        self.unk = unk

    def __call__(self, tokenized):
        return [w if w in self.vocabulary else self.unk for w in tokenized]


class ToTensor(object):
    def __init__(self, device='cpu', dtype=torch.long):
        self.device = device
        self.dtype = dtype

    def __call__(self, x):
        return mktensor(x, device=self.device, dtype=self.dtype)

