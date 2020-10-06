import argparse
import random

import numpy as np
import tqdm

import constants
from slp.util.system import find_substring_occurences


def read_word_misspellings(corpus):
    misspellings = {}
    with open(corpus, "r") as fd:
        for line in fd:
            tgt, src = line.strip().split("\t")
            misspellings[src] = misspellings.get(src, []) + [tgt]

    return misspellings


def read_sentences(corpus):
    with open(corpus, "r") as fd:
        sentences = [l.strip() for l in fd]

    return sentences


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
        if s[idx - 1] in constants.CHARACTERS and s[idx + 1] in constants.CHARACTERS:
            # only mess up spaces around words

            if random.random() < ratio:
                neighbor = idx + np.random.choice([-2, -1, 1, 2])
                noisy_sentence = (
                    noisy_sentence[:neighbor] + " " + noisy_sentence[neighbor:]
                )
                noisy_sentence = noisy_sentence[:idx] + noisy_sentence[idx + 1 :]

    return noisy_sentence


def make_sentence_noise_fn(word_misspellings, word_ratio=0.6, spacing_ratio=0.1):
    def sentence_noise(sentence):
        sentence = word_sentence_noise(sentence, word_misspellings, ratio=word_ratio)
        sentence = mess_up_spacing(sentence, ratio=spacing_ratio)

        return sentence

    return sentence_noise


def create_sentence_corpus(
    sentence_corpus,
    word_misspelling_corpus,
    output_file,
    keep_original=0.9,
    word_ratio=0.6,
    spacing_ratio=0.1,
    num_iter=10,
):
    word_misspellings = read_word_misspellings(word_misspelling_corpus)
    sentences = read_sentences(sentence_corpus)
    insert_noise = make_sentence_noise_fn(
        word_misspellings, word_ratio=word_ratio, spacing_ratio=spacing_ratio
    )
    with open(output_file, "w") as fd:
        for s in tqdm.tqdm(sentences):
            if random.random() < keep_original:
                src, tgt = s, s
                fd.write("{}\t{}\n".format(src, tgt))

            for _ in range(num_iter):
                src, tgt = insert_noise(s), s
                fd.write("{}\t{}\n".format(src, tgt))


def parse_args():
    parser = argparse.ArgumentParser("Generate sentence misspelling corpus")
    parser.add_argument("--corpus", type=str, help="Base sentence corpus file")
    parser.add_argument("--misspellings", type=str, help="Word misspelling corpus")
    parser.add_argument("--output", type=str, help="Output sentence corpus file")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    sentence_corpus = args.corpus
    word_misspelling_corpus = args.misspellings
    output_file = args.output
    create_sentence_corpus(sentence_corpus, word_misspelling_corpus, output_file)
