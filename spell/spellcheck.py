import argparse
import torch
import torch.nn as nn
import constants

from sklearn.metrics import accuracy_score
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.optim import Adam
from slp.config import SPECIAL_TOKENS

from slp.data.spelling import SpellCorrectorDataset
from slp.data.transforms import CharacterTokenizer
from slp.data.collators import Sequence2SequenceCollator
from slp.modules.seq2seq import Seq2SeqRNN
from slp.util.parallel import DataParallelModel, DataParallelCriterion


DEBUG = False


config = {
    "device": "cuda",
    "parallel": False,
    "num_workers": 1,
    "batch_size": 512,
    "lr": 1e-3,
    "epochs": 10,
    "hidden_size": 256,
    "embedding_size": 256,
    "tie_decoder": True,
    "encoder_layers": 2,
    "decoder_layers": 2,
    "encoder_dropout": .2,
    "decoder_dropout": .2,
    "rnn_type": "gru",
    "attention": False,
    "teacher_forcing": .4
}


if DEBUG:
    config["device"] = "cuda"
    config["batch_size"] = 128
    config["parallel"] = False
    config["num_workers"] = 0


def parse_args():
    parser= argparse.ArgumentParser("Train spell checker")
    parser.add_argument("--train", type=str, help="Train split file")
    parser.add_argument("--val", type=str, help="Validation split file")
    args=parser.parse_args()
    return args


collate_fn = Sequence2SequenceCollator(device='cpu')


def train_epoch(model, optimizer, criterion, train_loader, device='cpu', parallel=True):
    model = model.train()
    avg_loss, nbatch = 0, len(train_loader)
    n_correct, n_tokens = 0., 0.
    for bxi, batch in enumerate(tqdm(train_loader), 1):
        source, target, source_lengths = map(lambda x: x.to(device), batch)
        decoded = model(source, target, source_lengths)

        if parallel:
            loss = criterion([d.view(-1, d.size(-1)) for d in decoded], target.view(-1))
            loss = loss.mean()
            gathered = nn.parallel.gather(decoded, "cuda:0")
        else:
            loss = criterion(decoded.view(-1, decoded.size(-1)), target.view(-1))
            gathered = decoded
        dec_tok = gathered.argmax(-1)
        y_hat, y = dec_tok[target != 0].view(-1), target[target != 0].view(-1)
        n_correct += (y_hat == y).sum()
        n_tokens += len(y_hat)
        avg_loss += loss.item()
        if bxi % 100 == 0:
            print("Train iteration: {} \t Loss: {} \t Acc: {}".format(bxi, avg_loss / bxi, n_correct / n_tokens))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss = avg_loss / nbatch
    return avg_loss


def eval_epoch(model, criterion, val_loader, device='cpu', parallel=True):
    model = model.eval()
    avg_loss, nbatch = 0, len(val_loader)
    n_correct, n_tokens = 0., 0.
    with torch.no_grad():
        for batch in tqdm(val_loader):
            source, target, source_lengths = map(lambda x: x.to(device), batch)
            decoded = model(source, target, source_lengths)
            if parallel:
                loss = criterion([d.view(-1, d.size(-1)) for d in decoded], target.view(-1))
                loss = loss.mean()
                gathered = nn.parallel.gather(decoded, "cuda:0")
            else:
                loss = criterion(decoded.view(-1, decoded.size(-1)), target.view(-1))
                gathered = decoded
            dec_tok = gathered.argmax(-1)
            y_hat, y = dec_tok[target != 0].view(-1), target[target != 0].view(-1)
            n_correct += (y_hat == y).sum()
            n_tokens += len(y_hat)
            avg_loss += loss.item()
        avg_loss = avg_loss / nbatch
        accuracy = n_correct / n_tokens
    return avg_loss, accuracy


def train(model, optimizer, criterion, train_loader, val_loader, epochs=50, device='cpu', parallel=True):
    for e in range(epochs):
        _ =  train_epoch(model, optimizer, criterion, train_loader, device=device, parallel=parallel)
        train_loss, train_acc = eval_epoch(model, criterion, train_loader, device=device, parallel=parallel)
        print("Epoch: {}\tTrain loss: {}\tTrain accuracy: {}".format(e, train_loss, train_acc))
        val_loss, val_acc = eval_epoch(model, criterion, val_loader, device=device, parallel=parallel)
        print("Epoch: {}\tVal loss: {}\tVal accuracy: {}".format(e, val_loss, val_acc))
        torch.save(model.state_dict(), "spell_check.model.{}.pth".format(e))
        torch.save(optimizer.state_dict(), "spell_check.opt.{}.pth".format(e))


if __name__ == "__main__":
    args = parse_args()

    if DEBUG:
        args.train = "hnc.test"
        args.val = "hnc.test"
    tokenizer = CharacterTokenizer(
        constants.CHARACTER_VOCAB,
        prepend_bos=True,
        append_eos=True,
        specials=SPECIAL_TOKENS
    )

    sos_idx = tokenizer.c2i[SPECIAL_TOKENS.BOS.value]
    eos_idx = tokenizer.c2i[SPECIAL_TOKENS.EOS.value]

    vocab_size = len(tokenizer.vocab)

    trainset = SpellCorrectorDataset(args.train, tokenizer=tokenizer)
    valset = SpellCorrectorDataset(args.val, tokenizer=tokenizer)

    train_loader = DataLoader(
        trainset, batch_size=config["batch_size"], num_workers=config["num_workers"],
        pin_memory=True, shuffle=True, collate_fn=collate_fn, drop_last=True
    )
    val_loader = DataLoader(
        valset, batch_size=config["batch_size"], num_workers=config["num_workers"],
        pin_memory=True, shuffle=True, collate_fn=collate_fn, drop_last=True
    )

    model = Seq2SeqRNN(
        config["hidden_size"], vocab_size, vocab_size,
        embedding_size=config["embedding_size"], tie_decoder=config["tie_decoder"],
        encoder_layers=config["encoder_layers"], decoder_layers=config["decoder_layers"],
        encoder_dropout=config["encoder_dropout"], decoder_dropout=config["decoder_dropout"],
        rnn_type=config["rnn_type"], packed_sequence=True, attention=config["attention"],
        sos=sos_idx, eos=eos_idx, teacher_forcing_p=config["teacher_forcing"]
    )

    optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=config["lr"])
 
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    if config["parallel"]:
        model = DataParallelModel(model)
        criterion = DataParallelCriterion(criterion)
    model = model.to(config["device"])
    criterion = criterion.to(config["device"])

    train(model, optimizer, criterion, train_loader, val_loader, epochs=config["epochs"], device=config["device"], parallel=config["parallel"])
