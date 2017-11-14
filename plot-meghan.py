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
    parser.add_argument('-n', '--n-fits', type=int, default=5, **d)
    par_opts = parser.add_mutually_exclusive_group()
    par_opts.add_argument('-s', '--save-pars')
    par_opts.add_argument('-l', '--load-pars')
    par_opts.add_argument('-m', '--signal-multiplier', type=float,
                          default=1, nargs='?', const=5)
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
        sig_label = f'sig * {sm:.3g} + bg' if sm != 1.0 else 'sig + bg'
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
    def __call__(self, amp, decay, length, power, sub, p0, p1, p2):
        kernel = get_kernel(amp, decay, length, power, sub)
        mean = Mean((p0,p1,p2))
        gp = george.GP(kernel, mean=mean, fit_mean = True)
        try:
            gp.compute(self.x, self.yerr)
            return -gp.log_likelihood(self.y)
        except:
            return np.inf

GP_PARS = ['amp', 'decay', 'length', 'power', 'sub']
ALL_PARS = GP_PARS + FIT_PARS

def fit_gp_minuit(num, lnprob):
    from iminuit import Minuit
    from scipy.optimize import minimize

    min_likelihood = np.inf
    best_fit_params = (0, 0, 0, 0, 0)
    guesses = {
        'amp': 5701461179.0,
        'p0': 0.2336137363271451,
        'p1': 0.46257148564598083,
        'p2': 0.8901612464370032
    }
    def get_bound(par, neg=False):
        if neg:
            return (-2.0*guesses[par], 2.0*guesses[par])
        return (guesses[par] * 0.5, guesses[par] * 2.0)
    for i in range(num):
        init = {
            'amp': np.random.random() * 2*guesses['amp'],
            'decay': np.random.random() * 0.64,
            'length': np.random.random() * 5e5,
            'power': np.random.random() * 1.0,
            'sub': np.random.random() * 1.0,
            'p0': np.random.random() * guesses['p0'] * 2,
            'p1': np.random.random() * guesses['p1'] * 2,
            'p2': np.random.random() * guesses['p2'] * 2
        }
        bound = dict(
            amp = get_bound('amp'),
            decay = (10, 1000),
            length = (100, 1e8),
            power = (0.01, 1000),
            sub = (0.01, 1e6),
            p0 = get_bound('p0', neg=False),
            p1 = get_bound('p1', neg=True),
            p2 = get_bound('p2', neg=True)
        )

        def lnprob2(x):
            return lnprob(*x)

        result = minimize(
            lnprob2, [init[x] for x in ALL_PARS],
            bounds=[bound[x] for x in ALL_PARS]).x

        minuit_pars = {x: init[x] for x in ALL_PARS}
        for par in GP_PARS:
            minuit_pars['error_' + par] = 1e1
        for par in FIT_PARS:
            minuit_pars['error_' + par] = 1e-2
        minuit_pars.update({'limit_' + x: bound[x] for x in ALL_PARS})
        m = Minuit(lnprob, throw_nan = True, pedantic = True,
                   print_level = 0, errordef = 0.5,
                   **minuit_pars)
        m.migrad()
        # print(m.fval, lnprob2(result))
        # print(m.values, result)
        new_llh = lnprob2(result)
        par_dict = {x: result[n] for n, x in enumerate(ALL_PARS)}
        if new_llh < min_likelihood and m.fval != 0.0:
            min_likelihood = new_llh
            best_fit_params = par_dict
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

def get_kernel(amp, decay, length, power, sub):
    return amp * MyDijetKernelSimp(a = decay, b = length, c = power, d = sub)

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
