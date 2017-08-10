#!/usr/bin/env python3

"""
Simple script to draw data distributions
"""

from argparse import ArgumentParser
from h5py import File
import numpy as np

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('input_file')
    return parser.parse_args()

def run():
    args = parse_args()
    from pygp.canvas import Canvas

    with File(args.input_file,'r') as h5file:
        x, y = get_xy_pts(h5file['Nominal']['mjj_Data_2015_3p57fb'])

    with Canvas('spectrum.pdf') as can:
        can.ax.plot(x, y, '.')
        can.ax.set_yscale('log')

def get_xy_pts(group):
    assert 'hist_type' in group.attrs
    vals = np.asarray(group['values'])
    edges = np.asarray(group['edges'])
    center = (edges[:-1] + edges[1:]) / 2
    return center, vals[1:-1]

if __name__ == '__main__':
    run()
# from matplotlib import pyplot
# pyplot.errorbar(x, y, yerr=yerr_hack, fmt='.', zorder=0)
# pyplot.fill_between(t, mu - std, mu + std, facecolor=(0, 1, 0, 0.5), zorder=5, label='err = 1')
# pyplot.plot(t, mu, '-b', linewidth=1, zorder=4)

# pyplot.fill_between(t, mu_crap - std_crap, mu_crap + std_crap, facecolor=(1, 0, 0, 0.5), zorder=5, label='sqrt error')
# pyplot.plot(t, mu_crap, '-r', linewidth=1, zorder=4)
# pyplot.legend()

# a = pyplot.gca()
# a.set_yscale('log')
# a.set_ylim(10e-3, a.get_ylim()[1])

# pyplot.savefig('test.pdf')
# pyplot.show()

# input('press any button')
