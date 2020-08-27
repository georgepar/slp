# %%

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset

from slp.config import SPECIAL_TOKENS
from slp.data.collators import TransformerCollator
from slp.data.vocab import create_vocab
from slp.modules.transformer import Transformer
from slp.data.transforms import ToTokenIds, ToTensor
from slp.trainer import TransformerTrainer

collate_fn = TransformerCollator(device="cpu")

torch.backends.cudnn.deterministic = True  # type: ignore
torch.backends.cudnn.benchmark = False  # type: ignore
torch.manual_seed(0)


# All tokens are different. Should get 100% accuracy
sentence = "The big brown fox jumps over the lazy dog".split(" ")

vocab = create_vocab(sentence, vocab_size=50, extra_tokens=SPECIAL_TOKENS.to_list())
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
        device="cpu",
    )


def test_transformer_output_size():
    model = create_model(hidden_size=32)
    inputs, targets, mask1, mask2 = next(iter(train_loader))
    preds = model(inputs, targets, source_mask=mask1, target_mask=mask2)
    assert preds.size() == (1, len(sentence) - 1, len(vocab))


def test_inner_layers_output_size():
    hidden_size = 32
    model = create_model(hidden_size=hidden_size)
    inputs, targets, mask1, mask2 = next(iter(train_loader))

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


def test_model_overfits_single_batch():
    model = create_model(hidden_size=32)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    trainer = TransformerTrainer(
        model,
        optimizer,
        checkpoint_dir="../../checkpoints",
        experiment_name="transformer_lm_test",
        metrics=None,
        patience=4,
        validate_every=1000,
        accumulation_steps=1,
        loss_fn=nn.CrossEntropyLoss(),
        non_blocking=True,
        device="cpu",
    )

    trainer.fit(train_loader, train_loader, epochs=100)

    model.eval()
    inputs, targets, m1, m2 = next(iter(train_loader))
    preds = model(inputs, targets, source_mask=m1, target_mask=m2)
    pred_tokens = preds.max(-1)[1]
    print(model)
    print(f"Targets={targets}")
    print(f"Predicted={pred_tokens}")
    assert torch.all(torch.eq(targets, pred_tokens))
