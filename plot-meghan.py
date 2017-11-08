#!/usr/bin/env python3

"""
Simple script to draw data distributions
"""

from argparse import ArgumentParser
from h5py import File
import numpy as np
import george
from george.kernels import MyDijetKernelSimp

def parse_args():
    parser = ArgumentParser(description=__doc__)
    d = dict(help='%(default)s')
    parser.add_argument('input_file')
    parser.add_argument('signal_file', nargs='?')
    parser.add_argument('-e', '--output-file-extension', default='.pdf')
    parser.add_argument('-n', '--n-fits', type=int, default=10, **d)
    return parser.parse_args()

FIT_PARS = ['p0','p1','p2']

def run():
    args = parse_args()
    ext = args.output_file_extension
    from pygp.canvas import Canvas

    with File(args.input_file,'r') as h5file:
        x, y, xerr, yerr = get_xy_pts(
            h5file['dijetgamma_g85_2j65']['Zprime_mjj_var'])

    valid_x = (x > 0) & (x < 1500)
    valid_y = y > 0
    valid = valid_x & valid_y
    x, y = x[valid], y[valid]
    xerr, yerr = xerr[valid], yerr[valid]

    lnProb = logLike_minuit(x, y, xerr)
    min_likelihood, best_fit = fit_gp_minuit(20, lnProb)

    fit_pars = [best_fit[x] for x in FIT_PARS]
    with Canvas(f'points{ext}') as can:
        can.ax.errorbar(x, y, yerr=yerr, fmt='.')
        can.ax.set_yscale('log')
    kargs = {x:y for x, y in best_fit.items() if x not in FIT_PARS}
    kernel_new = get_kernel(**kargs)
    print(kernel_new.get_parameter_names())
    gp_new = george.GP(kernel_new, mean=Mean(fit_pars), fit_mean = True)

    plot_gp(x, y, xerr, yerr, gp_new, ext=args.output_file_extension)

def plot_gp(x, y, xerr, yerr, gp_new, ext):
    from pygp.canvas import Canvas
    t = np.linspace(np.min(x), np.max(x), 500)
    gp_new.compute(x, yerr)
    mu, cov = gp_new.predict(y, t)
    mu_x, cov_x = gp_new.predict(y, x)
    signif = (y - mu_x) / np.sqrt(np.diag(cov_x) + yerr**2)
    fit_mean_smooth = gp_new.mean.get_value(t)
    std = np.sqrt(np.diag(cov))

    with Canvas(f'spectrum{ext}') as can:
        can.ax.errorbar(x, y, yerr=yerr, fmt='.')
        can.ax.set_yscale('log')
        # can.ax.set_ylim(1, can.ax.get_ylim()[1])
        can.ax.plot(t, mu, '-r')
        # can.ax.plot(t, fit_mean_smooth, '--b')
        can.ax.fill_between(t, mu - std, mu + std,
                            facecolor=(0, 1, 0, 0.5),
                            zorder=5, label='err = 1')
        can.ratio.stem(x, signif, markerfmt='.', basefmt=' ')
        can.ratio.axhline(0, linewidth=1, alpha=0.5)

# _________________________________________________________________
# stuff copied from Meghan

class logLike_minuit:
    def __init__(self, x, y, xerr):
        self.x = x
        self.y = y
        self.xerr = xerr
    def __call__(self, Amp, decay, length, power, sub, p0, p1, p2):
        kernel = get_kernel(Amp, decay, length, power, sub)
        mean = Mean((p0,p1,p2))
        gp = george.GP(kernel, mean=mean, fit_mean = True)
        try:
            gp.compute(self.x, np.sqrt(self.y))
            return -gp.lnlikelihood(self.y, self.xerr)
        except:
            return np.inf

def fit_gp_minuit(num, lnprob):
    from iminuit import Minuit

    min_likelihood = np.inf
    best_fit_params = (0, 0, 0, 0, 0)
    guesses = {
        'amp': 5701461179.0,
        'p0': 0.2336137363271451,
        'p1': 0.46257148564598083,
        'p2': 0.8901612464370032
    }
    def bound(par, neg=False):
        if neg:
            return (-2.0*guesses[par], 2.0*guesses[par])
        return (guesses[par] * 0.5, guesses[par] * 2.0)
    for i in range(num):
        iamp = np.random.random() * 2*guesses['amp']
        idecay = np.random.random() * 0.64
        ilength = np.random.random() * 5e5
        ipower = np.random.random() * 1.0
        isub = np.random.random() * 1.0
        ip0 = np.random.random() * guesses['p0'] * 2
        ip1 = np.random.random() * guesses['p1'] * 2
        ip2 = np.random.random() * guesses['p2'] * 2
        m = Minuit(lnprob, throw_nan = True, pedantic = True,
                   print_level = 0, errordef = 0.5,
                   Amp = iamp,
                   decay = idecay,
                   length = ilength,
                   power = ipower,
                   sub = isub,
                   p0 = ip0, p1 = ip1, p2 = ip2,
                   error_Amp = 1e1,
                   error_decay = 1e1,
                   error_length = 1e1,
                   error_power = 1e1,
                   error_sub = 1e1,
                   error_p0 = 1e-2, error_p1 = 1e-2, error_p2 = 1e-2,
                   limit_Amp = bound('amp'),
                   limit_decay = (0.01, 1000),
                   limit_length = (100, 1e8),
                   limit_power = (0.01, 1000),
                   limit_sub = (0.01, 1e6),
                   limit_p0 = bound('p0', neg=False),
                   limit_p1 = bound('p1', neg=True),
                   limit_p2 = bound('p2', neg=True))
        m.migrad()
        print(m.fval)
        if m.fval < min_likelihood and m.fval != 0.0:
            min_likelihood = m.fval
            best_fit_params = m.values
    print("min LL", min_likelihood)
    print(f'best fit params {best_fit_params}')
    return min_likelihood, best_fit_params

class Mean():
    def __init__(self, params):
        self.p0=params[0]
        self.p1=params[1]
        self.p2=params[2]
    def get_value(self, t):
        sqrts = 13000.
        p0, p1, p2 = self.p0, self.p1, self.p2
        # steps = np.append(np.diff(t), np.diff(t)[-1])
        # print(steps)
        vals = (p0 * (1.-t/sqrts)**p1 * (t/sqrts)**(p2))
        # print(vals)
        return vals

def get_kernel(Amp, decay, length, power, sub):
    return Amp * MyDijetKernelSimp(a = decay, b = length, c = power, d = sub)

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
