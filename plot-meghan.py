#!/usr/bin/env python3

"""
Simple script to draw data distributions
"""

from argparse import ArgumentParser
from h5py import File
import numpy as np
import george

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('input_file')
    parser.add_argument('-e', '--output-file-extension', default='.pdf')
    return parser.parse_args()


def run():
    args = parse_args()
    from pygp.canvas import Canvas

    with File(args.input_file,'r') as h5file:
        x, y = get_xy_pts(h5file['Nominal']['mjj_Data_2015_3p57fb'])

    valid_x = (x > 1200) & (x < 7000)
    valid_y = y > 0
    valid = valid_x & valid_y
    x, y = x[valid], y[valid]
    yerr = np.sqrt(y)
    xerr = np.diff(x)

    lnProb = logLike_minuit(x, y, xerr)
    min_likelihood, best_fit_gp = fit_gp_minuit(1, lnProb)
    Amp, lengthscale, p0, p1, p2 = best_fit_gp

    kernel_new = Kernel((Amp, lengthscale))
    gp_new = george.GP(kernel_new, mean=Mean((p0,p1,p2)), fit_mean = True)

    gp_new.compute(x, yerr)
    t = np.linspace(np.min(x), np.max(x), 500)
    mu, cov = gp_new.predict(y, t)
    std = np.sqrt(np.diag(cov))

    ext = args.output_file_extension
    with Canvas(f'spectrum{ext}') as can:
        can.ax.errorbar(x, y, yerr=yerr, fmt='.')
        can.ax.set_yscale('log')
        can.ax.plot(t, mu, '-r')
        can.ax.fill_between(t, mu - std, mu + std,
                            facecolor=(0, 1, 0, 0.5),
                            zorder=5, label='err = 1')

# _________________________________________________________________
# stuff copied from Meghan

class logLike_minuit:
    def __init__(self, x, y, xerr):
        self.x = x
        self.y = y
        self.xerr = xerr
    def __call__(self, Amp, lengthscale, p0, p1, p2):
        kernel = Kernel((Amp, lengthscale))
        mean = Mean((p0,p1,p2))
        gp = george.GP(kernel, mean=mean, fit_mean = True)
        try:
            gp.compute(self.x, np.sqrt(self.y))
            return -gp.lnlikelihood(self.y)
        except:
            return 0

def fit_gp_minuit(num, lnprob):
    from iminuit import Minuit

    min_likelihood = np.inf
    best_fit_params = (0, 0, 0, 0, 0)
    for i in range(num):
        init0 = np.random.random() * 1e10
        init1 = np.random.random() * 5e5
        init2 = np.random.random() * 1.
        init3 = np.random.random() * 1.
        init4 = np.random.random() * 1.
        m = Minuit(lnprob, throw_nan = False, pedantic = False,
                   print_level = 0, errordef = 0.5,
                   Amp = init0, lengthscale = init1,
                   p0 = init2, p1 = init3, p2 = init4,
                   error_Amp = 1e1, error_lengthscale = 1e1,
                   error_p0 = 1e-2, error_p1 = 1e-2, error_p2 = 1e-2,
                   limit_Amp = (0.01, 1e15), limit_lengthscale = (100, 1e8),
                   limit_p0 = (0,10),
                   limit_p1 = (-2e2, 2e2), limit_p2 = (-2e2,2e2))
        m.migrad()
        if m.fval < min_likelihood:
            min_likelihood = m.fval
            best_fit_params = m.args
    amp, ls, p0, p1, p2 = best_fit_params
    print("min LL", min_likelihood)
    print(f'best fit vals: amp {amp:e}, ls {ls:e}, ps {p0} {p1} {p2}')
    return min_likelihood, best_fit_params

class Mean():
    def __init__(self, params):
        self.p0=params[0]
        self.p1=params[1]
        self.p2=params[2]
    def get_value(self, t):
        sqrts = 13000.
        p0, p1, p2 = self.p0, self.p1, self.p2
        return (p0 * (1.-t/sqrts)**p1 * (t/sqrts)**(p2))*np.append(np.diff(t), np.diff(t)[-1])

def Kernel(hyperparams):
    from george.kernels import ExpSquaredKernel
    return hyperparams[0]*ExpSquaredKernel(hyperparams[1])

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
