"""Concatenate multiple CSV result files into one (with dedup)."""

import argparse
import csv


def concat_csv(files, output):
    rows = []
    seen = set()
    fieldnames = None

    for filepath in files:
        with open(filepath, newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            else:
                for col in reader.fieldnames:
                    if col not in fieldnames:
                        fieldnames.append(col)
            for row in reader:
                key = tuple(sorted(row.items()))
                if key not in seen:
                    seen.add(key)
                    rows.append(row)

    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} lignes (doublons ignorés) depuis {len(files)} fichiers -> {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate CSV result files")
    parser.add_argument("files", nargs="+", help="CSV files to concatenate")
    parser.add_argument("-o", "--output", default="resultats_concat.csv", help="Output file")
    args = parser.parse_args()
    concat_csv(args.files, args.output)
