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
    with open(filename) as f:
        return dict(Counter(chain.from_iterable(map(str.split, f))))


def merge_counts(x, y):
    return {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}


def count_parallel(filenames, n_jobs=32):
    counts = ParallelRunner(
        n_jobs=n_jobs, total=len(filenames)
    )(
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
    parser= argparse.ArgumentParser("Corpus frequency counter")
    parser.add_argument("--corpora", type=str, help="Path to corpora")
    parser.add_argument("--njobs", type=int, help="njobs")
    parser.add_argument("--output", type=str, help="Output pickle file")
    args=parser.parse_args()
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
        "top100k": dict(counts.most_common(100_000)),
        "top50k": dict(counts.most_common(50_000)),
        "top30k": dict(counts.most_common(30_000)),
        "top10k": dict(counts.most_common(10_000)),
    }
    pickle_dump(out, args.output)
