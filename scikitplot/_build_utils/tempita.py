#!/usr/bin/env python3
import os
import sys
import argparse

# XXX: If this import ever fails (does it really?), vendor either
# cython.tempita or numpy/npy_tempita.
from Cython import Tempita as tempita


def process_tempita(fromfile, outfile=None):
    """Process tempita templated file and write out the result.

    The template file is expected to end in `.c.in` or `.pyx.in`:
    E.g. processing `template.c.in` generates `template.c`.

    """
    # template = tempita.Template.from_filename(
    #     fromfile,
    #     encoding=sys.getdefaultencoding()
    # )
    with open(fromfile, "r", encoding="utf-8") as f:
        template_content = f.read()
    template = tempita.Template(template_content)
    content = template.substitute()

    with open(outfile, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "infile",
        type=str,
        help="Path to the input file"
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        help="Path to the output file"
    )
    parser.add_argument(
        "-i",
        "--ignore",
        type=str,
        help="An ignored input - may be useful to add a "
        "dependency between custom targets",
    )
    args = parser.parse_args()

    if not args.infile.endswith('.in'):
        raise ValueError(f"Unexpected extension: {args.infile}")

    if not args.outfile:
        raise ValueError("Missing `--outfile` argument to tempita.py")

    if os.path.isabs(args.outfile):
        raise ValueError("`--outfile` must relative to the current directory")

    outdir_abs = os.path.join(os.getcwd(), args.outfile)
    outfile = os.path.join(
        outdir_abs, os.path.splitext(os.path.split(args.infile)[1])[0]
    )

    process_tempita(args.infile, outfile)


if __name__ == "__main__":
    main()