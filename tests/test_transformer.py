# %%
import itertools

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from slp.config.nlp import SPECIAL_TOKENS
from slp.data.collators import Seq2SeqCollator
from slp.data.corpus import create_vocab
from slp.data.transforms import ToTensor, ToTokenIds
from slp.modules.transformer import Transformer
from slp.plbind import PLDataModuleFromCorpus, TransformerPLModule, make_trainer
from slp.util.pytorch import pad_mask, subsequent_mask

from warnings import simplefilter

simplefilter(action="ignore")


pl.utilities.seed.seed_everything(42)


collate_fn = Seq2SeqCollator(device="cpu")


# All tokens are different. Should get 100% accuracy
sentence = "The big brown fox jumps over the lazy dog".split(" ")

vocab = create_vocab([sentence], vocab_size=-1, special_tokens=SPECIAL_TOKENS)
vocab = dict(zip(vocab.keys(), itertools.count()))
to_token_ids = ToTokenIds(vocab)
to_tensor = ToTensor(device="cpu")


class DummyDataset(Dataset):
    def __init__(self):
        self.data = [(sentence[0:-1], sentence[1:])]

    def __len__(self):
        return 1

    def __getitem__(self, i):
        s, t = self.data[i]

        return to_tensor(to_token_ids(s)), to_tensor(to_token_ids(t))


train_loader = DataLoader(DummyDataset(), batch_size=1, collate_fn=collate_fn)


def create_model(hidden_size=32):
    return Transformer(
        vocab_size=len(vocab),
        max_length=len(sentence) - 1,
        num_layers=1,
        hidden_size=hidden_size,
        num_heads=4,
        inner_size=128,
    )


def test_transformer_output_size():
    model = create_model(hidden_size=32)
    inputs, targets, leni, lent = next(iter(train_loader))
    mask1 = pad_mask(leni, max_length=torch.max(leni)).unsqueeze(-2)
    mask2 = pad_mask(lent, max_length=torch.max(lent)).unsqueeze(-2)
    mask2 = mask2 * subsequent_mask(torch.max(lent)).to(mask2.device)
    preds = model(inputs, targets, source_mask=mask1, target_mask=mask2)
    assert preds.size() == (1, len(sentence) - 1, len(vocab))


def test_inner_layers_output_size():
    hidden_size = 32
    model = create_model(hidden_size=hidden_size)
    inputs, targets, leni, lent = next(iter(train_loader))
    mask1 = pad_mask(leni, max_length=torch.max(leni)).unsqueeze(-2)
    mask2 = pad_mask(lent, max_length=torch.max(lent)).unsqueeze(-2)
    mask2 = mask2 * subsequent_mask(torch.max(lent)).to(mask2.device)

    x = model.embed(inputs)
    assert x.size() == (1, len(sentence) - 1, hidden_size)
    x = model.pe(x)
    assert x.size() == (1, len(sentence) - 1, hidden_size)

    y = model.pe(model.embed(targets))
    assert y.size() == (1, len(sentence) - 1, hidden_size)

    e = model.transformer_block.encoder(x, attention_mask=mask1)
    assert e.size() == (1, len(sentence) - 1, hidden_size)

    el = model.transformer_block.encoder.encoder[0](x, attention_mask=mask1)
    assert el.size() == (1, len(sentence) - 1, hidden_size)

    s1 = model.transformer_block.encoder.encoder[0].l1(x, attention_mask=mask1)
    assert s1.size() == (1, len(sentence) - 1, hidden_size)

    s2 = model.transformer_block.encoder.encoder[0].l2(s1)
    assert s2.size() == (1, len(sentence) - 1, hidden_size)

    d = model.transformer_block.decoder(x, y, source_mask=mask1, target_mask=mask2)
    assert d.size() == (1, len(sentence) - 1, hidden_size)

    dl = model.transformer_block.decoder.decoder[0](
        x, y, source_mask=mask1, target_mask=mask2
    )
    assert dl.size() == (1, len(sentence) - 1, hidden_size)

    s3 = model.transformer_block.decoder.decoder[0].in_layer(y, attention_mask=mask2)
    assert s3.size() == (1, len(sentence) - 1, hidden_size)

    s4 = model.transformer_block.decoder.decoder[0].fuse_layer(
        e, s3, attention_mask=mask1
    )
    assert s4.size() == (1, len(sentence) - 1, hidden_size)

    s5 = model.transformer_block.decoder.decoder[0].out_layer(s4)
    assert s5.size() == (1, len(sentence) - 1, hidden_size)

    o = model.transformer_block(x, y, source_mask=mask1, target_mask=mask2)
    assert o.size() == (1, len(sentence) - 1, hidden_size)
