import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchnlp.datasets import wikitext_2_dataset  # type: ignore

import pytorch_lightning as pl

from torchnlp.samplers import BPTTBatchSampler
from slp.util.log import configure_logging
from slp.config.nlp import SPECIAL_TOKENS
from slp.data import TransformerCollator


from slp.modules.transformer import Transformer
from slp.plbind import (
    PLDataModuleFromCorpus,
    FromLogits,
    TransformerPLModule,
    make_trainer,
    watch_model,
)


collate_fn = TransformerCollator(device="cpu")


if __name__ == "__main__":
    EXPERIMENT_NAME = "transformer-wikitext2"
    configure_logging(f"logs/{EXPERIMENT_NAME}")

    bptt = 35  # TODO: argparse this
    vocab_size = -1
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

    # train = train[:1000]
    # dev = dev[:1000]
    # test = test[:1000]

    ldm = PLDataModuleFromCorpus(
        train,
        val=dev,
        test=test,
        drop_last=True,
        max_len=-1,
        batch_sampler_train=BPTTBatchSampler(train, bptt, 20, True),
        batch_sampler_val=BPTTBatchSampler(dev, bptt, 10, True),
        batch_sampler_test=BPTTBatchSampler(test, bptt, 10, True),
        pin_memory=True,
        num_workers=0,
        language_model=True,
        tokens="tokenized",
        collate_fn=collate_fn,
    )

    model = Transformer(
        vocab_size=ldm.vocab_size,
        max_length=bptt,
        num_layers=2,
        hidden_size=200,
        num_heads=2,
        inner_size=256,
        dropout=0.2,
    )

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    criterion = nn.CrossEntropyLoss()

    lm = TransformerPLModule(
        model,
        optimizer,
        criterion,
        metrics={"acc": FromLogits(pl.metrics.classification.Accuracy())},
    )

    trainer = make_trainer(
        EXPERIMENT_NAME, max_epochs=100, gpus=1, gradient_clip_val=0.25
    )
    watch_model(trainer, model)

    trainer.fit(lm, datamodule=ldm)

    trainer.test(ckpt_path="best", test_dataloaders=ldm.test_dataloader())
