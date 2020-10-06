import argparse
import os

import torch
import torch.nn as nn
import transformers
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import constants
from slp.config import SPECIAL_TOKENS
from slp.data.collators import Sequence2SequenceCollator
from slp.data.spelling import \
    OnlineSpellCorrectorDataset as SpellCorrectorDataset
from slp.data.transforms import CharacterTokenizer, WordpieceTokenizer
from slp.modules.convs2s import Seq2Seq
from slp.util import log
from slp.util.parallel import DataParallelCriterion, DataParallelModel
from slp.util.system import safe_mkdirs

DEBUG = False


config = {
    "logfile": "spellchecker.online.fixed.log",
    "schedule": "linear",
    "checkpoint_dir": "online_errors.linear",
    "device": "cuda",
    "parallel": True,
    "num_workers": 2,
    "batch_size": 300,
    "lr": 4e-4,
    "max_steps": 100000,
    "warmup_steps": 8000,
    "checkpoint_steps": 5000,
    "hidden_size": 512,
    "embedding_size": 256,
    "encoder_kernel_size": 3,
    "decoder_kernel_size": 3,
    "encoder_layers": 10,
    "decoder_layers": 10,
    "encoder_dropout": 0.3,
    "decoder_dropout": 0.3,
    "max_length": 80,
    "gradient_clip": 0.1,
    # "teacher_forcing": 0.4,
}


LOGGER = log.mklogger(filename=config["logfile"])


if DEBUG:
    config["device"] = "cpu"
    config["batch_size"] = 128
    config["parallel"] = False
    config["num_workers"] = 0


def parse_args():
    parser = argparse.ArgumentParser("Train spell checker")
    parser.add_argument("--train", type=str, help="Train split file")
    parser.add_argument("--val", type=str, help="Validation split file")
    parser.add_argument("--word-errors", type=str, help="Word error corpus")
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


def step(model, source, target, parallel=False, device="cpu"):
    decoded = model(source, target[:, :-1])
    target = target[:, 1:]

    if parallel:
        loss = criterion(
            [d.contiguous().view(-1, d.size(-1)) for d in decoded],
            target.contiguous().view(-1),
        )
        loss = loss.mean()
        gathered = nn.parallel.gather(decoded, "cuda:0")
    else:
        loss = criterion(
            decoded.contiguous().view(-1, decoded.size(-1)),
            target.contiguous().view(-1),
        )
        gathered = decoded

    predicted = gathered.argmax(-1)

    return loss, predicted, target


def eval_epoch(model, criterion, val_loader, device="cpu", parallel=True):
    model = model.eval()

    running_loss = RunningLoss()
    running_acc = RunningAccuracy()

    with torch.no_grad():
        for bxi, batch in enumerate(tqdm(val_loader), 1):

            source, target, _ = map(lambda x: x.to(device), batch)
            loss, predicted, target = step(
                model, source, target, parallel=parallel, device=device
            )

            running_acc.push(predicted, target)
            running_loss.push(loss.item())

            if bxi % 100 == 0:
                LOGGER.info(
                    "Train iteration: {} \t Loss: {} \t Acc: {}".format(
                        bxi, running_loss.get(), running_acc.get()
                    )
                )

    avg_loss, accuracy = running_loss.get(), running_acc.get()

    return avg_loss, accuracy


def train_iterations(
    model,
    optimizer,
    scheduler,
    criterion,
    train_loader,
    val_loader,
    checkpoint_dir="./",
    max_steps=100000,
    checkpoint_steps=5000,
    device="cpu",
    parallel=True,
):
    safe_mkdirs(checkpoint_dir)
    model = model.train()
    clip = config["gradient_clip"]

    running_loss = RunningLoss()
    running_acc = RunningAccuracy()

    train_generator = iter(train_loader)

    for bxi in tqdm(range(1, max_steps + 1), total=max_steps):
        optimizer.zero_grad()
        try:
            batch = next(train_generator)
        except StopIteration:
            train_generator = iter(train_loader)
            batch = next(train_generator)

        source, target, _ = map(lambda x: x.to(device), batch)
        loss, predicted, target = step(
            model, source, target, parallel=parallel, device=device
        )

        running_acc.push(predicted, target)
        running_loss.push(loss.item())

        if bxi % 100 == 0:
            LOGGER.info(
                f"Train iteration: {bxi} \t Loss: {running_loss.get()} \t Acc: {running_acc.get()} \t LR: {optimizer.param_groups[0]['lr']}"
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        scheduler.step()

        if bxi % checkpoint_steps == 0:
            model = model.eval()
            val_loss, val_acc = eval_epoch(
                model, criterion, val_loader, device=device, parallel=parallel
            )
            LOGGER.info(
                "Step: {}\tVal loss: {}\tVal accuracy: {}".format(
                    bxi, val_loss, val_acc
                )
            )
            model_path = os.path.join(checkpoint_dir, f"spchk.model.{bxi}.pth")
            opt_path = os.path.join(checkpoint_dir, f"spchk.opt.{bxi}.pth")
            torch.save(model.state_dict(), model_path)
            torch.save(optimizer.state_dict(), opt_path)
            model = model.train()

    return running_loss.get()


if __name__ == "__main__":
    args = parse_args()

    if DEBUG:
        args.train = "hnc.test"
        args.val = "hnc.test"

    # tokenizer = WordpieceTokenizer(
    #     lower=True,
    #     bert_model="nlpaueb/bert-base-greek-uncased-v1",
    #     prepend_bos=True,
    #     append_eos=True,
    #     specials=SPECIAL_TOKENS,
    # )

    tokenizer = CharacterTokenizer(
        constants.CHARACTER_VOCAB,
        prepend_bos=True,
        append_eos=True,
        specials=SPECIAL_TOKENS,
    )

    sos_idx = tokenizer.c2i[SPECIAL_TOKENS.BOS.value]
    pad_idx = tokenizer.c2i[SPECIAL_TOKENS.PAD.value]
    eos_idx = tokenizer.c2i[SPECIAL_TOKENS.EOS.value]

    vocab_size = len(tokenizer.vocab)

    trainset = SpellCorrectorDataset(
        args.train,
        tokenizer=tokenizer,
        max_length=config["max_length"],
        word_misspelling_corpus=args.word_errors,
    )

    valset = SpellCorrectorDataset(
        args.val,
        tokenizer=tokenizer,
        max_length=config["max_length"],
        word_misspelling_corpus=args.word_errors,
    )

    train_loader = DataLoader(
        trainset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        valset,
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

    optimizer = transformers.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["lr"],
        weight_decay=0.01,
        # correct_bias=False,
    )

    if config["schedule"] == "linear":
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, config["warmup_steps"], config["max_steps"]
        )
    else:
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer, config["warmup_steps"], config["max_steps"]
        )

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    if config["parallel"]:
        model = DataParallelModel(model)
        criterion = DataParallelCriterion(criterion)
    model = model.to(config["device"])
    criterion = criterion.to(config["device"])

    train_iterations(
        model,
        optimizer,
        scheduler,
        criterion,
        train_loader,
        val_loader,
        checkpoint_dir=config["checkpoint_dir"],
        max_steps=config["max_steps"],
        checkpoint_steps=config["checkpoint_steps"],
        device=config["device"],
        parallel=config["parallel"],
    )
