#!/usr/bin/env python3

import numpy as np

# Generate some fake noisy data.
x = np.linspace(0, 100, 101)
bg_ideal = 40 * np.exp(-x*0.05)
bg = np.fromiter( (np.random.poisson(y) for y in bg_ideal), dtype=float)
y = bg
yerr_hack = np.ones(bg_ideal.shape)
yerr = np.sqrt(bg_ideal)

import george
from george.kernels import ExpSquaredKernel

# Set up the Gaussian process.
kernel = ExpSquaredKernel(10.0)
gp = george.GP(kernel)

# Pre-compute the factorization of the matrix.
gp.compute(x, yerr_hack)

# Compute the log likelihood.
print(gp.lnlikelihood(y))


t = np.linspace(0, np.max(x), 500)
mu, cov = gp.predict(y, t)
std = np.sqrt(np.diag(cov))

# less hacky gp
gp_crap = george.GP(kernel)
gp_crap.compute(x, yerr)
mu_crap, cov_crap = gp_crap.predict(y, t)
std_crap = np.sqrt(np.diag(cov_crap))


# draw results
from matplotlib import pyplot
pyplot.errorbar(x, y, yerr=yerr_hack, fmt='.', zorder=0)
pyplot.fill_between(t, mu - std, mu + std, facecolor=(0, 1, 0, 0.5), zorder=5, label='err = 1')
pyplot.plot(t, mu, '-b', linewidth=1, zorder=4)

pyplot.fill_between(t, mu_crap - std_crap, mu_crap + std_crap, facecolor=(1, 0, 0, 0.5), zorder=5, label='sqrt error')
pyplot.plot(t, mu_crap, '-r', linewidth=1, zorder=4)
pyplot.legend()

a = pyplot.gca()
a.set_yscale('log')
a.set_ylim(10e-3, a.get_ylim()[1])

pyplot.savefig('test.pdf')
pyplot.show()

input('press any button')
