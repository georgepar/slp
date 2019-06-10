import spacy
import torch

from pytorch_pretrained_bert import BertTokenizer
from spacy.attrs import ORTH


class WordpieceTokenizer(object):
    def __init__(self, lower=True, bert_model='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model,
                                                       do_lower_case=lower)
        self.tokenizer.max_len = 1024  # hack to suppress warnings

    def __call__(self, x):
        x = ['[CLS]'] + self.tokenizer.tokenize(x)
        return self.tokenizer.convert_tokens_to_ids(x)


class SpacyTokenizer(object):
    def __init__(self, lower=True, bos='<bos>', eos='<eos>',
                 specials=['<pad>', '<unk>', '<mask>', '<bos>', '<eos>']):
        self.lower = lower
        self.nlp = self.get_nlp(specials=specials)
        self.bos = bos
        self.eos = eos

    def get_nlp(name="en_core_web_sm", specials=['<pad>', '<unk>',
                                                 '<mask>', '<bos>', '<eos>']):
        nlp = spacy.load(name)
        for control_token in specials:
            nlp.tokenizer.add_special_case(
                control_token, [{ORTH: control_token}])
        return nlp

    def __call__(self, x):
        if self.lower:
            x = x.lower()
        x = [y.text for y in self.nlp.tokenizer(x)]
        if self.bos:
            x = [self.bos] + x
        if self.eos:
            x = x + [self.eos]
        return x


class ToTokensSpacy(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, x):
        return [self.vocab.stoi[w]
                if w != '<unk>' else self.vocab.stoi['<oov>']
                for w in x]


class ToTensor(object):
    def __init__(self, device='cpu', dtype=torch.long):
        self.device = device
        self.dtype = dtype

    def __call__(self, x):
        return torch.tensor(x, device=self.device, dtype=self.dtype)
