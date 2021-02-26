import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchnlp.datasets import wikitext_2_dataset  # type: ignore

import pytorch_lightning as pl

from slp import configure_logging
from slp.config import SPECIAL_TOKENS
from slp.data import (
    LMDataset,
    TransformerCollator,
    create_vocab,
    ReplaceUnknownToken,
    ToTokenIds,
    ToTensor,
)


from slp.modules.transformer import Transformer
from slp.plbind import (
    PLDataModuleFromDatasets,
    FromLogits,
    TransformerPLModule,
    make_trainer,
    watch_model,
)


collate_fn = TransformerCollator(device="cpu")


if __name__ == "__main__":
    EXPERIMENT_NAME = "transformer-wikitext2"
    configure_logging(f"logs/{EXPERIMENT_NAME}")

    max_len = 80  # TODO: argparse this
    vocab_size = 20000
    lr = 1e-4

    train, dev, test = wikitext_2_dataset(
        directory="data/",
        train=True,
        dev=True,
        test=True,
        extracted_name="wikitext-2",
        url="https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip",  # noqa: E501
        unknown_token=SPECIAL_TOKENS.UNK.value,
        eos_token=SPECIAL_TOKENS.EOS.value,
    )

    train = train
    dev = dev
    test = test

    vocab = create_vocab(
        train + dev, vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS
    )
    vocab = dict(zip(vocab.keys(), itertools.count()))
    replace_unk = ReplaceUnknownToken()
    to_token_ids = ToTokenIds(vocab)
    to_tensor = ToTensor(device="cpu")

    def create_dataset(base):
        return (
            LMDataset(base, max_len=max_len)
            .map(replace_unk)
            .map(to_token_ids)
            .map(to_tensor)
            .apply_transforms()
        )

    train, dev, test = map(create_dataset, [train, dev, test])

    ldm = PLDataModuleFromDatasets(
        train,
        val=dev,
        test=test,
        batch_size=64,
        batch_size_eval=128,
        drop_last=True,
        collate_fn=collate_fn,
    )

    model = Transformer(
        vocab_size=len(vocab),
        max_length=max_len,
        num_layers=2,
        hidden_size=128,
        num_heads=4,
        inner_size=256,
    )

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    criterion = nn.CrossEntropyLoss()

    lm = TransformerPLModule(
        model,
        optimizer,
        criterion,
        metrics={"acc": FromLogits(pl.metrics.classification.Accuracy())},
    )

    trainer = make_trainer(EXPERIMENT_NAME, max_epochs=100, gpus=1)
    watch_model(trainer, model)

    trainer.fit(lm, datamodule=ldm)

    trainer.test(ckpt_path="best", test_dataloaders=ldm.test_dataloader())
