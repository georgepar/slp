import itertools
from torch.utils.data import DataLoader
from torchnlp.datasets import wikitext_2_dataset

from slp.config import SPECIAL_TOKENS
from slp.data.datasets import LMDataset
from slp.data.collators import TransformerCollator
from slp.data.transforms import ToTokenIds, ToTensor


collate_fn = TransformerCollator(device='cpu')


if __name__ == "__main__":
    max_len = 256  # TODO: argparse this
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
        set(train + dev + list(map(lambda x: x.value, SPECIAL_TOKENS)))),
        itertools.count()))
    to_token_ids = ToTokenIds(vocab)
    to_tensor = ToTensor(device='cpu')

    def create_dataloader(base):
        wrapped = (LMDataset(base, max_len=max_len)
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