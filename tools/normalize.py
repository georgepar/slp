import glob
import os
import sys
import unicodedata

from joblib import Parallel, delayed
from tqdm.auto import tqdm


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


n_jobs = 4
filenames = []
for path in sys.argv[1:]:
    filenames += glob.glob(os.path.join(path, "*"))


def _is_punctuation(char):
  """Checks whether `chars` is a punctuation character."""
  cp = ord(char)
  # We treat all non-letter/number ASCII as punctuation.
  # Characters such as "^", "$", and "`" are not in the Unicode
  # Punctuation class but we treat them as punctuation anyways, for
  # consistency.
  if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
      (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
    return True
  cat = unicodedata.category(char)
  if cat.startswith("P"):
    return True
  return False


def _run_split_on_punc(text):
    """Splits punctuation on a piece of text."""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
        char = chars[i]
        if _is_punctuation(char):
            output.append([char])
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)
        i += 1

    return ["".join(x) for x in output]

def strip_accents_and_lowercase(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn').lower()

def normalize(filename):
    output_file = open(filename + ".out", 'w', encoding='utf8')
    with open(filename + ".encoded", encoding='utf8') as file:
        for line in file.readlines():
            tokens = line.lower().split()
            splited_tokens = []
            for token in tokens:
                splited_tokens.extend(_run_split_on_punc(token))
            line = ' '.join(splited_tokens)
            line = strip_accents_and_lowercase(line)
            if line.endswith('\n'):
                output_file.write(line)
            else:
                output_file.write(line+'\n')
    output_file.close()


def process_file(filename):
    lines = []
    with open(filename) as file:
        for line in file.readlines():
            lines.append(line.encode("utf-8", "ignore").decode())

    with open(filename + ".encoded", 'w') as file:
        for line in lines:
            if line.endswith('\n'):
                line = line.strip('\n')
            file.write(line + '\n')


if __name__ == '__main__':
    ProgressParallel(
        n_jobs=n_jobs, total=len(filenames)
    )(
        delayed(process_file)(fname) for fname in filenames
    )
    ProgressParallel(
        n_jobs=n_jobs, total=len(filenames)
    )(
        delayed(normalize)(fname) for fname in filenames
    )
