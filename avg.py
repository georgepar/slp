import glob
import os
import sys
import numpy as np


if __name__ == "__main__":
    results_dir = sys.argv[1]
    results_files = glob.glob(os.path.join(results_dir, "*"))
    res = {}

    for rf in results_files:
        with open(rf, "r") as fd:
            results = [l.strip().split(":") for l in fd]
            for metric, value in results:
                value = float(value.strip())
                if metric in res:
                    res[metric].append(value)
                else:
                    res[metric] = [value]

    for m, values in res.items():

        print("{}:\t{}+-{}".format(m, np.mean(values), np.std(values)))
