#!/usr/bin/env python3

"""
Simple script to draw signal points
"""

from argparse import ArgumentParser
from h5py import File
import numpy as np
from pygp.canvas import Canvas

def parse_args():
    parser = ArgumentParser(description=__doc__)
    d = dict(help='%(default)s')
    parser.add_argument('input_bg_file')
    parser.add_argument('signal_file')
    parser.add_argument('-e', '--output-file-extension', default='.pdf')
    return parser.parse_args()

def run():
    args = parse_args()
    ext = args.output_file_extension

    spectra = {}

    with File(args.input_bg_file,'r') as h5file:
        spectra['background'] = get_xy_pts(
            h5file['dijetgamma_g85_2j65']['Zprime_mjj_var'])

    with File(args.signal_file,'r') as h5file:
        spectra['signal'] = get_xy_pts(
            h5file['dijetgamma_g85_2j65']['Zprime_mjj_var'])

    valid_spectra = {}
    for name, (x, y, xerr, yerr) in spectra.items():
        valid_x = (x > 0) & (x < 1500)
        valid = valid_x
        x, y = x[valid], y[valid]
        xerr, yerr = xerr[valid], yerr[valid]
        valid_spectra[name] = (x, y, xerr, yerr)

    with Canvas(f'sig-vs-bg{ext}') as can:
        x, y, xerr, yerr = valid_spectra['background']
        x_sig, y_sig, xerr_sig, yerr_sig = valid_spectra['signal']
        can.ax.errorbar(x, y, yerr=yerr, fmt='.')
        can.ax.errorbar(x_sig, y_sig + y, yerr=yerr_sig, fmt='.')
        can.ax.set_yscale('log')
        valid = (y_sig > 0) & (y > 0)
        signif = y_sig[valid] / np.sqrt(yerr_sig[valid]**2 + yerr[valid]**2)
        can.ratio.stem(x[valid], signif, markerfmt='.', basefmt=' ')
        can.ratio.axhline(0, linewidth=1, alpha=0.5)
        can.ratio.set_xlabel(r'Der $m_{jj}$', ha='right', x=0.98)


def get_xy_pts(group):
    assert 'hist_type' in group.attrs
    vals = np.asarray(group['values'])
    edges = np.asarray(group['edges'])
    errors = np.asarray(group['errors'])
    center = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)
    return center, vals[1:-1], widths, errors[1:-1]


if __name__ == '__main__':
    run()
