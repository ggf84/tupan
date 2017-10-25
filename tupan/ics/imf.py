# -*- coding: utf-8 -*-
#

"""
TODO.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.integrate import quad
from scipy.optimize import fminbound
from ..units import ureg


def plot_imf(m, pdf, name):
    counts, edges = np.histogram(
        np.log10(m.m_as('M_sun')),
        bins='auto',
        # density=True,
    )

    x = edges[1:] + edges[:-1]
    x /= 2
    x = 10.0**x
    plt.step(x, counts, where='mid')

    n = len(m) * np.diff(edges)
    x = np.logspace(-2.5, +2.5, len(n))
    plt.plot(x, n * np.log(10.0) * x * pdf(x * ureg.M_sun), '--', label=name)

    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='best')
    plt.xlabel(r'$m\,\left[M_{\odot}\right]$')
    plt.ylabel(r'$\frac{d\,N(\log\,m)}{d\,\log\,m}$')
    plt.tight_layout()
    plt.show()


class IMFSample(object):
    """

    """
    def __init__(self, imf_func, mmin, mmax, name=''):
        """

        """
        self.mmin = mmin
        self.mmax = mmax
        self.__name__ = name

        norm = 1
        if mmin != mmax:
            norm, _ = quad(lambda m: imf_func(m * ureg.M_sun),
                           mmin.m_as('M_sun'),
                           mmax.m_as('M_sun'))

        self.pdf = lambda m: imf_func(m) / norm

        mpeak = fminbound(lambda m: -self.pdf(m * ureg.M_sun),
                          mmin.m_as('M_sun'), mmax.m_as('M_sun'),
                          xtol=1.0e-8) * ureg.M_sun
        self.fpeak = self.pdf(mpeak)

    def sample(self, n, seed=None):
        np.random.seed(seed)

        m = np.zeros(0) * ureg.M_sun
        nm = len(m)
        while nm < n:
            x = np.exp(np.random.uniform(np.log(self.mmin.m_as('M_sun')),
                                         np.log(self.mmax.m_as('M_sun')),
                                         size=n-nm)) * ureg.M_sun
            y = np.random.uniform(0.0, self.fpeak, size=n-nm)
            g = self.pdf(x) * x.m_as('M_sun')
            m = np.concatenate([m, x.compress(y < g)]) * ureg.M_sun
            nm = len(m)

        plot_imf(m, self.pdf, self.__name__)

        return m, m.sum()


class IMF(object):
    """

    """
    @staticmethod
    def equalmass():
        mmin = 1.0 * ureg.M_sun
        mmax = 1.0 * ureg.M_sun

        def imf_func(m):
            return m.m_as('M_sun')**0.0

        return IMFSample(imf_func, mmin, mmax, 'equalmass')

    @staticmethod
    def salpeter1955(mmin=0.1 * ureg.M_sun,
                     mmax=120.0 * ureg.M_sun):

        def imf_func(m):
            return m.m_as('M_sun')**(-2.35)

        return IMFSample(imf_func, mmin, mmax, 'salpeter1955')

    @staticmethod
    def padoan2007(mmin=0.004 * ureg.M_sun,
                   mmax=120.0 * ureg.M_sun):
        m_ch = 1.0 * ureg.M_sun
        sigma = 1.8
        gamma = 1.4

        def imf_func(m):
            f = 1 + erf((4 * np.log(m/m_ch) + sigma**2) / (np.sqrt(8) * sigma))
            return f * m.m_as('M_sun')**(-gamma-1)

        return IMFSample(imf_func, mmin, mmax, 'padoan2007')

    @staticmethod
    def parravano2011(mmin=0.004 * ureg.M_sun,
                      mmax=120.0 * ureg.M_sun):
        mu = 0.35 * ureg.M_sun
        gamma = 1.35

        def imf_func(m):
            f = 1 - np.exp(-(m / mu)**(0.51 + gamma))
            return f * m.m_as('M_sun')**(-gamma-1)

        return IMFSample(imf_func, mmin, mmax, 'parravano2011')

    @staticmethod
    def maschberger2013(mmin=0.01 * ureg.M_sun,
                        mmax=150.0 * ureg.M_sun):
        class IMFSample2(object):
            def __init__(self, mu, alpha, beta, name='maschberger2013'):
                self.mu = mu
                self.alpha = alpha
                self.beta = beta
                self.__name__ = name

            def sample(self, n, seed=None):
                np.random.seed(seed)
                i = np.arange(n+1)
                u = np.random.uniform(i+0, i+1, size=n+1) / (n+1)
                u = np.random.permutation(u[:n])

                mu, alpha, beta = self.mu, self.alpha, self.beta
                Gmmin = pow(1 + pow(mmin/mu, 1-alpha), 1-beta)
                Gmmax = pow(1 + pow(mmax/mu, 1-alpha), 1-beta)

                u *= Gmmax - Gmmin
                u += Gmmin
                m = mu * pow(pow(u, 1/(1-beta)) - 1, 1/(1-alpha))

                def pdf(m):
                    k = (1 - alpha) * (1 - beta) / mu.m_as('M_sun')
                    k /= Gmmax - Gmmin
                    return (k * pow(m/mu, -alpha)
                              * pow(1 + pow(m/mu, 1-alpha), -beta))

                plot_imf(m, pdf, self.__name__)

                return m, m.sum()

        return IMFSample2(0.2 * ureg.M_sun, 2.3, 1.4)


# -- End of File --
