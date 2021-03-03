import math

import torch.nn as nn
import torch.optim as optim
from torchnlp.datasets import wikitext_2_dataset  # type: ignore
from torchnlp.samplers import BPTTBatchSampler

from slp.config.nlp import SPECIAL_TOKENS
from slp.data import Seq2SeqCollator
from slp.modules.embed import PositionalEncoding
from slp.modules.transformer import Encoder as TransformerEncoder
from slp.modules.transformer import Transformer
from slp.plbind import (
    PLDataModuleFromCorpus,
    TransformerPLModule,
    make_trainer,
    watch_model,
)
from slp.util.log import configure_logging


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size=30000,
        num_layers=2,
        hidden_size=200,
        num_heads=2,
        inner_size=256,
        dropout=0.2,
        tie_weights=True,
    ):
        super(TransformerLM, self).__init__()
        self.pos_encoder = PositionalEncoding(hidden_size, max_len=5000)
        self.transformer_encoder = TransformerEncoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            inner_size=inner_size,
            dropout=dropout,
        )
        self.hidden_size = hidden_size
        self.encoder = nn.Embedding(vocab_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.tie_weights = tie_weights

        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)

        if not self.tie_weights:
            nn.init.zeros_(self.decoder.weight)
            nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, targets, source_mask=None, target_mask=None):
        src = self.encoder(src) * math.sqrt(self.hidden_size)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, attention_mask=target_mask)
        output = self.decoder(output)

        return output


collate_fn = Seq2SeqCollator(device="cpu")


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
        tokenizer="tokenized",
        collate_fn=collate_fn,
    )

    model = TransformerLM(
        vocab_size=ldm.vocab_size,
        num_layers=2,
        hidden_size=200,
        num_heads=2,
        inner_size=256,
        dropout=0.2,
        tie_weights=True,
    )

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    criterion = nn.CrossEntropyLoss()

    lm = TransformerPLModule(
        model,
        optimizer,
        criterion,
        calculate_perplexity=True,
    )

    trainer = make_trainer(
        EXPERIMENT_NAME, max_epochs=100, gpus=1, gradient_clip_val=0.25
    )
    watch_model(trainer, model)

    trainer.fit(lm, datamodule=ldm)

    trainer.test(ckpt_path="best", test_dataloaders=ldm.test_dataloader())
