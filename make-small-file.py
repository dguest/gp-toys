#!/usr/bin/env python3

from h5py import File
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    return parser.parse_args()

COPY_DIRS = [
    ('dijetgamma_g85_2j65', 'Zprime_mjj_var')
]

def run():
    args = parse_args()
    with File(args.input_file, 'r') as in_file:
        with File(args.output_file, 'w') as out_file:
            for d1, d2, in COPY_DIRS:
                grp = out_file.require_group(d1)
                if d2 not in grp:
                    in_file.copy(f'{d1}/{d2}', grp)

if __name__ == '__main__':
    run()
