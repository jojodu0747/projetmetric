"""Concatenate two CSV result files into one."""

import argparse
import csv
import sys


def concat_csv(file1: str, file2: str, output: str):
    rows = []
    fieldnames = None

    for filepath in [file1, file2]:
        with open(filepath, newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            else:
                # Merge headers (in case columns differ slightly)
                for col in reader.fieldnames:
                    if col not in fieldnames:
                        fieldnames.append(col)
            for row in reader:
                rows.append(row)

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} lignes -> {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate two CSV result files")
    parser.add_argument("file1", help="First CSV file")
    parser.add_argument("file2", help="Second CSV file")
    parser.add_argument("-o", "--output", default="resultats_concat.csv", help="Output file (default: resultats_concat.csv)")
    args = parser.parse_args()
    concat_csv(args.file1, args.file2, args.output)
