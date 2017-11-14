#!/usr/bin/env python3

"""
Simple script to draw data distributions
"""

from argparse import ArgumentParser
from h5py import File
import numpy as np
import george
from george.kernels import MyDijetKernelSimp
import json

def parse_args():
    parser = ArgumentParser(description=__doc__)
    d = dict(help='%(default)s')
    parser.add_argument('input_file')
    parser.add_argument('signal_file', nargs='?')
    parser.add_argument('-e', '--output-file-extension', default='.pdf')
    parser.add_argument('-n', '--n-fits', type=int, default=10, **d)
    par_opts = parser.add_mutually_exclusive_group()
    par_opts.add_argument('-s', '--save-pars')
    par_opts.add_argument('-l', '--load-pars')
    par_opts.add_argument('-m', '--signal-multiplier', type=float, default=1)
    return parser.parse_args()

FIT_PARS = ['p0','p1','p2']
FIT_RANGE = (0, 1500)

def run():
    args = parse_args()
    ext = args.output_file_extension
    from pygp.canvas import Canvas

    with File(args.input_file,'r') as h5file:
        x, y, xerr, yerr = get_xy_pts(
            h5file['dijetgamma_g85_2j65']['Zprime_mjj_var'],
            FIT_RANGE)

    if args.load_pars:
        with open(args.load_pars, 'r') as pars_file:
            best_fit = json.load(pars_file)
    else:
        lnProb = logLike_minuit(x, y, yerr)
        _, best_fit = fit_gp_minuit(20, lnProb)
    if args.save_pars:
        with open(args.save_pars, 'w') as pars_file:
            json.dump(best_fit, pars_file, indent=2)

    fit_pars = [best_fit[x] for x in FIT_PARS]
    with Canvas(f'points{ext}') as can:
        can.ax.errorbar(x, y, yerr=yerr, fmt='.')
        can.ax.set_yscale('log')
    kargs = {x:y for x, y in best_fit.items() if x not in FIT_PARS}
    kernel_new = get_kernel(**kargs)
    gp_new = george.GP(kernel_new, mean=Mean(fit_pars), fit_mean = True)

    ext = args.output_file_extension
    gp_new.compute(x, yerr)
    plot_gp(x, y, xerr, yerr, gp_new, name=f'spectrum{ext}')

    if not args.signal_file:
        return

    with File(args.signal_file,'r') as h5file:
        x_sig, y_sig, xerr_sig, yerr_sig = get_xy_pts(
            h5file['dijetgamma_g85_2j65']['Zprime_mjj_var'],
            FIT_RANGE)
        y_sig *= args.signal_multiplier
        yerr_sig *= args.signal_multiplier
        tot_error = (yerr**2 + yerr_sig**2)**(1/2)
        plot_gp(x_sig, y + y_sig, xerr_sig, tot_error, gp_new,
                name=f'with-signal{ext}', y_bg=y, yerr_bg=yerr,
                signal_multiplier=args.signal_multiplier)

def plot_gp(x, y, xerr, yerr, gp_new, name, y_bg=None, yerr_bg=None,
            signal_multiplier=1.0):
    from pygp.canvas import Canvas
    t = np.linspace(np.min(x), np.max(x), 500)
    mu, cov = gp_new.predict(y, t)
    mu_x, cov_x = gp_new.predict(y, x)
    signif = (y - mu_x) / np.sqrt(np.diag(cov_x) + yerr**2)
    fit_mean_smooth = gp_new.mean.get_value(t)
    std = np.sqrt(np.diag(cov))

    with Canvas(name) as can:
        sm = signal_multiplier
        sig_label = f'sig * {sm} + bg' if sm != 1.0 else 'sig + bg'
        can.ax.errorbar(x, y, yerr=yerr, fmt='.', label=sig_label)
        if y_bg is not None:
            can.ax.errorbar(x, y_bg, yerr=yerr_bg, fmt='.', label='bg')
        can.ax.set_yscale('log')
        can.ax.plot(t, mu, '-r')
        # can.ax.plot(t, fit_mean_smooth, '-k', label='fit function')
        can.ax.fill_between(t, mu - std, mu + std,
                            facecolor=(0, 1, 0, 0.5),
                            zorder=5, label=r'GP error = $1\sigma$')
        can.ax.legend(framealpha=0)
        can.ax.set_ylabel('events')
        can.ratio.stem(x, signif, markerfmt='.', basefmt=' ')
        can.ratio.axhline(0, linewidth=1, alpha=0.5)
        can.ratio.set_xlabel(r'$m_{jj}$ [GeV]', ha='right', x=0.98)
        can.ratio.set_ylabel('significance')

# _________________________________________________________________
# stuff copied from Meghan

class logLike_minuit:
    def __init__(self, x, y, yerr):
        self.x = x
        self.y = y
        self.yerr = yerr
    def __call__(self, Amp, decay, length, power, sub, p0, p1, p2):
        kernel = get_kernel(Amp, decay, length, power, sub)
        mean = Mean((p0,p1,p2))
        gp = george.GP(kernel, mean=mean, fit_mean = True)
        try:
            gp.compute(self.x, self.yerr)
            return -gp.lnlikelihood(self.y)
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

def get_xy_pts(group, x_range=None):
    assert 'hist_type' in group.attrs
    vals = np.asarray(group['values'])
    edges = np.asarray(group['edges'])
    errors = np.asarray(group['errors'])
    center = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)

    if x_range is not None:
        low, high = x_range
        ok = (center > low) & (center < high)

    return center[ok], vals[1:-1][ok], widths[ok], errors[1:-1][ok]

if __name__ == '__main__':
    run()
