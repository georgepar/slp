import argparse
import glob
import os

from collections import Counter
from itertools import chain
from joblib import delayed
from tqdm.auto import tqdm

from slp.util.multicore import ParallelRunner
from slp.util.system import pickle_dump


def get_filenames(corpora_path):
    filenames = glob.glob(os.path.join(corpora_path, "*"))
    return filenames


def count_file(filename):
    char_counts = {}
    with open(filename) as fd:
        for line in fd:
            for c in line.strip():
                char_counts[c] = char_counts.get(c, 0) + 1
    return Counter(char_counts)


def merge_counts(x, y):
    return {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}


def count_parallel(filenames, n_jobs=32):
    counts = ParallelRunner(n_jobs=n_jobs, total=len(filenames))(
        delayed(count_file)(fname) for fname in filenames
    )
    merged = {}
    for cnt in tqdm(counts):
        merged = merge_counts(merged, cnt)
    return Counter(merged)


def filter_counts(counts, thres=10):
    filt = {}
    for k, v in counts.items():
        if v > thres:
            filt[k] = v
    return filt


def parse_args():
    parser = argparse.ArgumentParser("Corpus frequency counter")
    parser.add_argument("--corpora", type=str, help="Path to corpora")
    parser.add_argument("--njobs", type=int, help="njobs")
    parser.add_argument("--output", type=str, help="Output pickle file")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    counts = count_parallel(get_filenames(args.corpora), n_jobs=args.njobs)
    print(counts.most_common(50))
    out = {
        "all": dict(counts),
        ">1": filter_counts(counts, thres=1),
        ">10": filter_counts(counts, thres=10),
        ">100": filter_counts(counts, thres=100),
        ">200": filter_counts(counts, thres=200),
        ">500": filter_counts(counts, thres=500),
        ">1000": filter_counts(counts, thres=1000),
        "top30": dict(counts.most_common(30)),
        "top50": dict(counts.most_common(50)),
        "top100": dict(counts.most_common(100)),
        "top200": dict(counts.most_common(200)),
    }
    pickle_dump(out, args.output)
