import argparse

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import constants
from slp.config import SPECIAL_TOKENS
from slp.data.collators import Sequence2SequenceCollator
from slp.data.spelling import SpellCorrectorDataset
from slp.data.transforms import CharacterTokenizer, WordpieceTokenizer
from slp.modules.convs2s import Seq2Seq

DEBUG = False


class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

config = {
    "device": "cuda",
    "parallel": True,
    "num_workers": 2,
    "batch_size": 128,
    "lr": 1e-3,
    "max_steps": 200000,
    "schedule_steps": 1000,
    "checkpoint_steps": 10000,
    "hidden_size": 512,
    "embedding_size": 256,
    "encoder_kernel_size": 3,
    "decoder_kernel_size": 3,
    "encoder_layers": 10,
    "decoder_layers": 10,
    "encoder_dropout": 0.3,
    "decoder_dropout": 0.3,
    "max_length": 100,
    "gradient_clip": 0.1,
    # "teacher_forcing": 0.4,
}


if DEBUG:
    config["device"] = "cpu"
    config["batch_size"] = 128
    config["num_workers"] = 0


def parse_args():
    parser = argparse.ArgumentParser("Evaluate spell checker")
    parser.add_argument("--test", type=str, help="Test split file")
    parser.add_argument("--ckpt", type=str, help="Checkpoint")
    args = parser.parse_args()

    return args


collate_fn = Sequence2SequenceCollator(device="cpu")


class RunningLoss(object):
    def __init__(self):
        self.n_items = 0
        self.accumulator = 0

    def push(self, value):
        self.accumulator += value
        self.n_items += 1

    def get(self):
        return self.accumulator / self.n_items if self.n_items > 0 else 0


class RunningAccuracy(object):
    def __init__(self):
        self.n_total = 0
        self.n_correct = 0

    def push(self, predicted, target):
        y_hat, y = predicted[target != 0].view(-1), target[target != 0].view(-1)
        self.n_correct += (y_hat == y).sum().item()
        self.n_total += len(y_hat)

    def get(self):
        return self.n_correct / self.n_total if self.n_total > 0 else 0


def step(model, source, target, device="cpu"):
    decoded = model(source, target[:, :-1])
    target = target[:, 1:]

    loss = criterion(
        decoded.contiguous().view(-1, decoded.size(-1)),
        target.contiguous().view(-1),
    )
    gathered = decoded

    predicted = gathered.argmax(-1)

    return loss, predicted, target


def eval_epoch(model, criterion, val_loader, device="cpu"):
    model = model.eval()

    running_loss = RunningLoss()
    running_acc = RunningAccuracy()

    with torch.no_grad():
        for bxi, batch in enumerate(tqdm(val_loader), 1):

            source, target, _ = map(lambda x: x.to(device), batch)
            loss, predicted, target = step(model, source, target, device=device)

            running_acc.push(predicted, target)
            running_loss.push(loss.item())

            if bxi % 100 == 0:
                print(
                    "Train iteration: {} \t Loss: {} \t Acc: {}".format(
                        bxi, running_loss.get(), running_acc.get()
                    )
                )

    avg_loss, accuracy = running_loss.get(), running_acc.get()

    return avg_loss, accuracy


if __name__ == "__main__":
    args = parse_args()

    # tokenizer = CharacterTokenizer(
    #     constants.CHARACTER_VOCAB,
    #     prepend_bos=True,
    #     append_eos=True,
    #     specials=SPECIAL_TOKENS,
    # )
    tokenizer = WordpieceTokenizer(
        lower=True,
        bert_model="nlpaueb/bert-base-greek-uncased-v1",
        prepend_bos=True,
        append_eos=True,
        specials=SPECIAL_TOKENS,
    )

    sos_idx = 1  # tokenizer.c2i[SPECIAL_TOKENS.BOS.value]
    pad_idx = 0  # tokenizer.c2i[SPECIAL_TOKENS.PAD.value]
    eos_idx = 2  # tokenizer.c2i[SPECIAL_TOKENS.EOS.value]

    vocab_size = len(tokenizer.vocab)

    testset = SpellCorrectorDataset(
        args.test, tokenizer=tokenizer, max_length=config["max_length"]
    )

    test_loader = DataLoader(
        testset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    model = Seq2Seq(
        vocab_size,
        vocab_size,
        hidden_size=config["hidden_size"],
        embedding_size=config["embedding_size"],
        encoder_layers=config["encoder_layers"],
        decoder_layers=config["decoder_layers"],
        encoder_kernel_size=config["encoder_kernel_size"],
        decoder_kernel_size=config["decoder_kernel_size"],
        encoder_dropout=config["encoder_dropout"],
        decoder_dropout=config["decoder_dropout"],
        max_length=config["max_length"],
        device=config["device"],
        pad_idx=pad_idx,
        # teacher_forcing_p=config["teacher_forcing"],
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model = WrappedModel(model)
    state_dict = torch.load(args.ckpt)
    model.load_state_dict(state_dict)
    model = model.to(config["device"])
    criterion = criterion.to(config["device"])

    test_loss, test_acc = eval_epoch(
        model,
        criterion,
        test_loader,
        device=config["device"],
    )
    print("Loss: {}\tAccuracy: {}".format(test_loss, test_acc))
