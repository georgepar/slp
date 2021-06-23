import os
import numpy as np
import argparse
import csv
from collections import Counter, OrderedDict


def average_column(csv_filepath):
    column_totals = {}
    with open(csv_filepath) as f:
        reader = csv.reader(f)
        header = next(reader)
        # print(f"Headers are {header}")
        for h in header:
            column_totals[h] = []
        row_count = 0.0
        for row in reader:
            for column_idx, column_value in enumerate(row):
                try:
                    n = float(column_value)
                    column_totals[header[column_idx]].append(n)
                except ValueError:
                    print(
                        f"Error -- ({column_value}) Column({column_idx}) could "
                        "not be converted to float!"
                    )
            row_count += 1.0
    # print(column_totals)
    # row_count is now 1 too many so decrement it back down
    # row_count -= 1.0

    # make sure column index keys are in order
    # column_indexes = column_totals.keys()
    # column_indexes.sort()
    # import pdb; pdb.set_trace()
    # calculate per column averages using a list comprehension
    average_totals = {h: 0.0 for h in header}
    std_totals = {h: 0.0 for h in header}
    for h in header:
        average_totals[h] = np.mean(column_totals[h])
        std_totals[h] = np.std(column_totals[h])
        # column_totals[h] = column_totals[h] / row_count
        # column_totals[h] = column_totals[h] / row_count
    # averages = [column_totals[h]/row_count for h in header]
    return average_totals, std_totals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="calculate average metrics")
    parser.add_argument("--path_to_csv", type=str, required=True)
    args = parser.parse_args()

    path_to_csv = args.path_to_csv

    avg, std = average_column(path_to_csv)

    print(f"The average per col is \n")
    for k, v in avg.items():
        print(k, v)

    print(f"The std per col is \n")
    for k, v in std.items():
        print(k, v)
