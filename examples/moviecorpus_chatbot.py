import numpy as np
import torch
import torch.nn as nn

from ignite.metrics import Loss
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.transforms import Compose

from slp.util.embeddings import EmbeddingsLoader
from slp.data.moviecorpus import MovieCorpusDataset
from slp.data.transforms import SpacyTokenizer, ToTokenIds, ToTensor
from slp.data.collators import Seq2SeqCollator
from slp.trainer.trainer import Seq2SeqTrainer
from slp.config.moviecorpus import SPECIAL_TOKENS

from slp.modules.seq2seq import EncoderDecoder, EncoderLSTM, DecoderLSTM_v2

from torch.optim import Adam


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
COLLATE_FN = Seq2SeqCollator(device='cpu')
MAX_EPOCHS = 50


def dataloaders_from_indices(dataset, train_indices, val_indices, batch_train,
                             batch_val):
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_train,
        sampler=train_sampler,
        drop_last=False,
        collate_fn=COLLATE_FN)
    val_loader = DataLoader(
        dataset,
        batch_size=batch_val,
        sampler=val_sampler,
        drop_last=False,
        collate_fn=COLLATE_FN)
    return train_loader, val_loader


def train_test_split(dataset, batch_train, batch_val,
                     test_size=0.2, shuffle=True, seed=None):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    test_split = int(np.floor(test_size * dataset_size))
    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)

    train_indices = indices[test_split:]
    val_indices = indices[:test_split]
    return dataloaders_from_indices(dataset, train_indices, val_indices,
                                    batch_train, batch_val)


def trainer_factory(embeddings, pad_index, device=DEVICE):
    encoder = EncoderLSTM(
        hidden_size=254, embeddings=embeddings, device=DEVICE)
    decoder = DecoderLSTM_v2(
        max_target_len=14, output_size=43, hidden_size=254,
        embeddings=embeddings, device=DEVICE)

    model = EncoderDecoder(
        encoder, decoder, teacher_forcing_ratio=.8, device=DEVICE)

    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_index)

    metrics = {
        'loss': Loss(criterion)
    }

    trainer = Seq2SeqTrainer(model,
                             optimizer,
                             checkpoint_dir=None, # '../checkpoints',
                             metrics=metrics,
                             non_blocking=True,
                             retain_graph=True,
                             patience=5,
                             device=device,
                             loss_fn=criterion)
    return trainer


if __name__ == '__main__':
    loader = EmbeddingsLoader(
        '../cache/glove.6B.50d.txt', 50, extra_tokens=SPECIAL_TOKENS)
    word2idx, idx2word, embeddings = loader.load()

    pad_index = idx2word[SPECIAL_TOKENS.PAD.value]

    tokenizer = SpacyTokenizer(prepend_bos=True,
                               append_eos=True,
                               specials=SPECIAL_TOKENS)
    to_token_ids = ToTokenIds(word2idx)
    to_tensor = ToTensor(device='cpu')

    transforms = Compose([tokenizer, to_token_ids, to_tensor])
    dataset = MovieCorpusDataset('./data/', transforms=transforms)

    train_loader, val_loader = train_test_split(dataset, 32, 128)
    trainer = trainer_factory(embeddings, pad_index, device=DEVICE)
    final_score = trainer.fit(train_loader, val_loader, epochs=MAX_EPOCHS)

    print(f'Final score: {final_score}')
