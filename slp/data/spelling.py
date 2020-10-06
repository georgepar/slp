import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from slp.data.transforms import ToTensor
from slp.util.system import find_substring_occurences

CHARACTERS = [
    "α",
    "β",
    "γ",
    "δ",
    "ε",
    "ζ",
    "η",
    "θ",
    "ι",
    "κ",
    "λ",
    "μ",
    "ν",
    "ξ",
    "ο",
    "π",
    "ρ",
    "σ",
    "τ",
    "υ",
    "φ",
    "χ",
    "ψ",
    "ω",
    "ς",
]


def word_sentence_noise(s, word_misspellings, ratio=0.5):
    noisy_sentence = ""

    for w in s.split(" "):
        noisy_sentence += " "

        if w in word_misspellings and random.random() < ratio:
            idx = random.randint(0, len(word_misspellings[w]) - 1)
            noisy_word = word_misspellings[w][idx]
            noisy_sentence += noisy_word
        else:
            noisy_sentence += w
    noisy_sentence = noisy_sentence.strip()

    return noisy_sentence


def mess_up_spacing(s, ratio=0.1):
    noisy_sentence = s
    space_idxes = find_substring_occurences(s, " ")

    for idx in space_idxes:
        if s[idx - 1] in CHARACTERS and s[idx + 1] in CHARACTERS:
            # only mess up spaces around words

            if random.random() < ratio:
                neighbor = idx + np.random.choice([-2, -1, 1, 2])
                noisy_sentence = (
                    noisy_sentence[:neighbor] + " " + noisy_sentence[neighbor:]
                )
                noisy_sentence = noisy_sentence[:idx] + noisy_sentence[idx + 1 :]

    return noisy_sentence


def make_sentence_noise_fn(word_misspellings, word_ratio=0.8, spacing_ratio=0.1):
    def sentence_noise(sentence):
        sentence = word_sentence_noise(sentence, word_misspellings, ratio=word_ratio)
        sentence = mess_up_spacing(sentence, ratio=spacing_ratio)

        return sentence

    return sentence_noise


def read_word_misspellings(corpus):
    misspellings = {}
    with open(corpus, "r") as fd:
        for line in fd:
            tgt, src = line.strip().split("\t")
            misspellings[src] = misspellings.get(src, []) + [tgt]

    return misspellings


class SpellCorrectorDataset(Dataset):
    def __init__(self, fname, tokenizer=None, max_length=256):
        self.data = []
        with open(fname, "r", errors="ignore") as fd:
            for ln in fd:
                try:
                    dat = ln.strip().split("\t")

                    if (
                        len(dat) == 2
                        and len(dat[1]) > 2
                        and len(dat[1]) < max_length
                        and len(dat[0]) < max_length
                    ):
                        self.data.append(dat)
                except:
                    pass
            # self.data = [line.strip().split("\t") for line in fd]
        print("Read {} lines".format(len(self.data)))
        self.tokenizer = tokenizer
        self.tt = ToTensor(device="cpu", dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]

        src = self.tt(self.tokenizer(src))
        tgt = self.tt(self.tokenizer(tgt))

        return src, tgt


class OnlineSpellCorrectorDataset(Dataset):
    def __init__(
        self, fname, tokenizer=None, max_length=256, word_misspelling_corpus=None
    ):
        self.data = []
        word_misspellings = read_word_misspellings(word_misspelling_corpus)
        self.sentence_noiser = make_sentence_noise_fn(word_misspellings)
        with open(fname, "r", errors="ignore") as fd:
            lines = fd.readlines()

            for i, ln in enumerate(lines):
                try:
                    dat = ln.strip()

                    if len(dat) <= 2:
                        continue

                    if len(dat) < max_length:
                        self.data.append(dat)
                    else:
                        words = dat.split(" ")
                        samples = []

                        current_sample = ""

                        while len(words) > 0:
                            if len(self.data) > 200000:
                                break
                            if len(current_sample + " " + words[0]) < max_length:
                                current_sample = (
                                    current_sample + " " + words.pop(0)

                                    if len(current_sample) > 0
                                    else words.pop(0)
                                )
                            else:
                                samples.append(current_sample)
                                current_sample = ""

                        if len(current_sample) > 0:
                            samples.append(current_sample)
                        self.data = self.data + samples

                except:
                    pass
            # self.data = [line.strip().split("\t") for line in fd]
        print("Read {} samples from {} lines".format(len(self.data), len(lines)))
        self.tokenizer = tokenizer
        self.tt = ToTensor(device="cpu", dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tgt = self.data[idx]

        src = self.tt(self.tokenizer(self.sentence_noiser(tgt)))
        tgt = self.tt(self.tokenizer(tgt))

        return src, tgt


class SpellCorrectorPredictionDataset(Dataset):
    def __init__(self, fname, tokenizer=None, max_length=256):
        self.data = []

        with open(fname, "r", errors="ignore") as fd:
            for ln in fd:
                try:
                    dat = ln.strip()
                    self.data.append(dat)
                except:
                    pass
        self.tokenizer = tokenizer
        self.tt = ToTensor(device="cpu", dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src = self.data[idx]
        src = self.tt(self.tokenizer(src))
        tgt = self.tt(self.tokenizer(""))

        return src, tgt
