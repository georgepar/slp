import itertools
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchnlp.datasets import wikitext_2_dataset

from slp.config import SPECIAL_TOKENS
from slp.data.datasets import LMDataset
from slp.data.collators import TransformerCollator
from slp.modules.transformer import Transformer
from slp.data.transforms import ReplaceUnknownToken, ToTokenIds, ToTensor
from slp.trainer import TransformerTrainer

collate_fn = TransformerCollator(device='cpu')


if __name__ == "__main__":
    max_len = 64  # TODO: argparse this
    vocab_size = 5000
    lr = 1e-4

    train, dev, test = wikitext_2_dataset(
        directory='data/',
        train=True,
        dev=True,
        test=True,
        extracted_name='wikitext-2',
        url='https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip',
        unknown_token=SPECIAL_TOKENS.UNK.value,
        eos_token=SPECIAL_TOKENS.EOS.value)
    vocab = dict(zip(list(
        set(SPECIAL_TOKENS.to_list() + train + dev)),
        itertools.count())[:vocab_size])
    replace_unk = ReplaceUnknownToken()
    to_token_ids = ToTokenIds(vocab)
    to_tensor = ToTensor(device='cpu')

    def create_dataloader(base):
        wrapped = (LMDataset(base, max_len=max_len)
                   .map(replace_unk)
                   .map(to_token_ids)
                   .map(to_tensor)
                   .apply_transforms())
        print(wrapped[0])
        return DataLoader(
            wrapped, batch_size=128,
            num_workers=1,
            pin_memory=True,
            collate_fn=collate_fn)

    train_loader = create_dataloader(train)
    dev_loader = create_dataloader(dev)
    test_loader = create_dataloader(test)

    model = Transformer(vocab_size=vocab_size,
                        max_length=max_len,
                        num_layers=2,
                        hidden_size=128,
                        num_heads=4,
                        inner_size=512)

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr)

    trainer = TransformerTrainer(
        model,
        optimizer,
        checkpoint_dir='../../checkpoints',
        experiment_name='transformer_lm',
        metrics=None,
        patience=4,
        validate_every=1,
        accumulation_steps=2,
        loss_fn=nn.CrossEntropyLoss(),
        non_blocking=True,
        device='cuda:0')

    trainer.fit(train_loader, dev_loader, epochs=10)
