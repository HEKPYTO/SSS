#!/usr/bin/env python3
"""trn2arff.py - convert an FST TRN data file to dense ARFF.

Used to produce the ARFF form of the paper's TRN-format datasets (madelon,
gisette) for the standalone sSFFS implementation, which reads ARFF only.
The TRN header gives per-class sample counts; the data section lists the
samples class by class. The output ARFF keeps that order (attributes
f1..fD numeric, nominal 'class' attribute c0..c{K-1} last).

Usage: python3 trn2arff.py input.trn output.arff
"""
import sys


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: trn2arff.py input.trn output.arff")
    features = classes = None
    class_sizes = []
    data_start = None
    # utf-8-sig: some TRN files carry a UTF-8 BOM; '#data' must be an exact
    # match ('#datafile' opens every TRN header and must not trigger it)
    with open(sys.argv[1], encoding="utf-8-sig") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith(";") or not s:
            continue
        if s.startswith("#features"):
            features = int(s.split()[1].replace(",", " "))
        elif s.startswith("#classes"):
            parts = s.replace(",", " ").split()
            classes = int(parts[1])
            class_sizes = [int(x) for x in parts[2:2 + classes]]
        elif s == "#data":
            data_start = i + 1
            break
        elif s.startswith("#"):
            continue
        elif features is not None and classes is not None and data_start is None:
            data_start = i  # data section without a #data keyword
            break
    if features is None or classes is None or data_start is None:
        sys.exit("trn2arff.py: malformed TRN header")

    values = " ".join(lines[data_start:]).split()
    total = sum(class_sizes)
    if len(values) < total * features:
        sys.exit(f"trn2arff.py: expected {total * features} values, found {len(values)}")

    with open(sys.argv[2], "w") as out:
        out.write(f"% converted from {sys.argv[1]} by trn2arff.py\n")
        out.write("@RELATION trn_converted\n\n")
        for j in range(features):
            out.write(f"@ATTRIBUTE f{j + 1} numeric\n")
        out.write("@ATTRIBUTE class {" + ",".join(f"c{c}" for c in range(classes)) + "}\n\n")
        out.write("@DATA\n")
        v = 0
        for c, size in enumerate(class_sizes):
            for _ in range(size):
                out.write(",".join(values[v:v + features]) + f",c{c}\n")
                v += features
    print(f"wrote {sys.argv[2]}: {features} features, {classes} classes, sizes {class_sizes}")


if __name__ == "__main__":
    main()
