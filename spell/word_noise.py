import argparse
import numpy as np
import random

import constants

from joblib import delayed

from slp.util.multicore import ParallelRunner
from slp.util.system import find_substring_occurences


def ignore_token(w):
    # Ignore word that doesn't contain a greek character
    dont_ignore = any(ch in constants.CHARACTERS for ch in w) and len(w) > 2
    return not dont_ignore


def insert_error_to_word(word, use_common_errors=False, bigram=False, error_type="ins"):
    if ignore_token(word):
        return word

    allowed = constants.CHARACTERS if not bigram else constants.BIGRAMS
    error = None

    def random_error(errors):
        chars = list(word)
        valid_errors = []
        for c1, c2 in errors:
            if c1 in word:
                valid_errors.append((c1, c2))
        error = random.choice(valid_errors) if len(valid_errors) > 0 else None
        return error

    if error_type == "del":
        bigram = False
        idx = random.randint(0, len(word) - 1)
        to_insert = ""
        continue_idx = idx + 1
    elif error_type == "ins":
        bigram = False
        if use_common_errors:
            error = random_error(constants.KEYBOARD_NEIGHBORS)
        if not use_common_errors or error is None:
            idx = random.randint(0, len(word) - 1)
            to_insert = allowed[np.random.randint(0, len(allowed))]
        else:
            idx = np.random.choice(find_substring_occurences(word, error[0]))
            to_insert = error[1]
        continue_idx = idx
    elif error_type == "sub":
        if use_common_errors:
            errors = constants.UNIGRAM_ERRORS if not bigram else constants.BIGRAM_ERRORS
            error = random_error(errors)
        if not use_common_errors or error is None:
            idx = random.randint(0, len(word) - 1)
            to_insert = allowed[np.random.randint(0, len(allowed))]
            continue_idx = idx + 1 if not bigram else idx + random.randint(1, 2)
        else:
            idx = np.random.choice(find_substring_occurences(word, error[0]))
            to_insert = error[1]
            continue_idx = idx + len(error[0])
    elif error_type == "shuf":
        idx = random.randint(0, len(word) - 2)
        to_insert = word[idx + 1] + word[idx]
        continue_idx = idx + 2
    else:
        raise ValueError("Unknown error type. Select: [ins|del|sub|shuf]")

    return word[:idx] + to_insert + word[continue_idx:]


def word_noise(word, num_errors=1, only_common_errors=False):
    num_errors = min(num_errors, max(len(word) - 4, 1))
    for _ in range(num_errors):
        error_type = np.random.choice(ERROR_TYPES)
        bigram = np.random.choice([True, False], p=[0.3, 0.7])
        if only_common_errors:
            use_common_errors = True
        else:
            if len(word) <= 4 and bigram:
                use_common_errors = True
            else:
                use_common_errors = np.random.choice([True, False])
        word = insert_error_to_word(
            word,
            use_common_errors=use_common_errors,
            bigram=bigram,
            error_type=error_type,
        )
    return word


def create_word_misspellings(word, iterations=100, only_common_errors=False):
    misspellings = []
    for _ in range(iterations):
        num_errors = np.random.choice([1, 2], p=[0.8, 0.2])
        noisy_word = word_noise(
            word, num_errors=num_errors, only_common_errors=only_common_errors
        )
        if noisy_word != word:
            misspellings.append((noisy_word, word))
    return list(set(misspellings))


def mkspellcorpus(words, n_jobs=32, only_common_errors=False):
    corpus = ParallelRunner(n_jobs=n_jobs, total=len(words))(
        delayed(create_word_misspellings)(
            word,
            iterations=MAX_MISSPELLINGS_PER_WORD,
            only_common_errors=only_common_errors,
        )
        for word in words
    )
    corpus = [el for sublist in corpus for el in sublist]
    return corpus


def parse_args():
    parser = argparse.ArgumentParser("Generate word misspelling corpus")
    parser.add_argument("--vocab", type=str, help="Vocabulary file")
    parser.add_argument("--njobs", type=int, help="njobs")
    parser.add_argument("--common", action="store_true", help="Use common errors only")
    parser.add_argument("--output", type=str, help="Output pickle file")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    vocab = args.vocab
    corpus_file = args.output
    only_common_errors = args.common
    with open(vocab, "r") as fd:
        words = [l.strip() for l in fd]

    corpus = mkspellcorpus(
        words, n_jobs=args.njobs, only_common_errors=only_common_errors
    )
    with open(corpus_file, "w") as fd:
        for src, tgt in corpus:
            fd.write("{}\t{}\n".format(src, tgt))


if __name__ == "__main__":
    main()
