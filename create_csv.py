import glob
import os
import sys
from tabulate import tabulate

if __name__ == "__main__":
    results_dirs = sys.argv[1:]

    all_results = {}

    for results_dir in results_dirs:
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

        metrics = list(res.keys())
        all_results[results_dir] = res

    lines = []
    header = ["Exp/Metrics"] + metrics

    for k, res in all_results.items():
        avgs = [sum(v) / len(v) for v in res.values()]
        lines.append([k] + [str(a) for a in avgs])

    print(tabulate(lines, headers=header, floatfmt=".3f"))
