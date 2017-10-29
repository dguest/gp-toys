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
    parser.add_argument('input_file')
    parser.add_argument('-e', '--output-file-extension', default='.pdf')
    return parser.parse_args()

FIT_PARS = ['p0','p1','p2']

def run():
    args = parse_args()
    ext = args.output_file_extension
    from pygp.canvas import Canvas

    with File(args.input_file,'r') as h5file:
        # h5file['Nominal']['mjj_Data_2015_3p57fb']
        # dijetgamma_g85_2j65 Zprime_mjj_var
        x, y = get_xy_pts(h5file['dijetgamma_g85_2j65']['Zprime_mjj_var'])

    valid_x = (x > 0) & (x < 2000)
    valid_y = y > 0
    valid = valid_x & valid_y
    x, y = x[valid], y[valid]
    yerr = np.sqrt(y)
    xerr = np.diff(x)


    lnProb = logLike_minuit(x, y, xerr)
    min_likelihood, best_fit = fit_gp_minuit(1, lnProb)

    t = np.linspace(np.min(x), np.max(x), 500)
    fit_pars = [best_fit[x] for x in FIT_PARS]
    fit_line = Mean(fit_pars).get_value(t)
    with Canvas(f'points{ext}') as can:
        can.ax.errorbar(x, y, yerr=yerr, fmt='.')
        can.ax.plot(t, fit_line, linestyle='-')
        can.ax.set_yscale('log')
    kargs = {x:y for x, y in best_fit.items() if x not in FIT_PARS}
    kernel_new = get_kernel(**kargs)
    gp_new = george.GP(kernel_new, mean=Mean(fit_pars), fit_mean = True)

    gp_new.compute(x, yerr)
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
    def __call__(self, Amp, decay, length, power, sub, p0, p1, p2):
        kernel = get_kernel(Amp, decay, length, power, sub)
        mean = Mean((p0,p1,p2))
        gp = george.GP(kernel, mean=mean, fit_mean = True)
        try:
            gp.compute(self.x, np.sqrt(self.y))
            return -gp.lnlikelihood(self.y, self.xerr)
        except:
            return 0

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
    def bound(par):
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
                   limit_p0 = bound('p0'),
                   limit_p1 = bound('p1'),
                   limit_p2 = bound('p2'))
        m.migrad()
        print(m.fval)
        if m.fval < min_likelihood:
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
        return (p0 * (1.-t/sqrts)**p1 * (t/sqrts)**(p2))*np.append(np.diff(t), np.diff(t)[-1])

def get_kernel(Amp, decay, length, power, sub):
    return Amp * MyDijetKernelSimp(a = decay, b = length, c = power, d = sub)

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
