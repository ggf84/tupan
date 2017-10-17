# -*- coding: utf-8 -*-
#

"""
TODO.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import fminbound


def plot_imf(m, ml, mu, name):
    import matplotlib.pyplot as plt
    bins = np.logspace(
        np.log10(ml),
        np.log10(mu),
        int(np.sqrt(len(m))),
    )
    plt.hist(
        m,
        log=True,
        label=name,
        bins=bins,
        histtype='step'
    )
    plt.xscale('log')
    plt.legend(loc='best')
    plt.xlabel(r'$m\,\left[M_{\odot}\right]$')
    plt.ylabel(r'$m\,p(m) = \frac{d\,N(\log\,m)}{d\,\log\,m}$')
    plt.tight_layout()
    plt.show()


class IMFSample(object):
    """

    """
    def __init__(self, imf_func, min_mlow, max_mhigh, mlow, mhigh, name=''):
        """

        """
        self.__name__ = name
        if mlow < min_mlow or mhigh > max_mhigh:
            raise ValueError('Mass range outer of limits'
                             ' [{0}:{1}].'.format(min_mlow, max_mhigh))

        def intarg(m):
            return imf_func(m)/m

        norm, _ = quad(intarg, min_mlow, max_mhigh)

        def imf_func_normed(m):
            return imf_func(m)/norm

        mpeak = float(fminbound(lambda m: -imf_func_normed(m),
                                min_mlow, max_mhigh,
                                xtol=1.0e-8))
        peak = imf_func_normed(mpeak)

        self.func = imf_func_normed
        self.peak = peak
        self.mlow = mlow
        self.mpeak = mpeak
        self.mhigh = mhigh
        self.min_mlow = min_mlow
        self.max_mhigh = max_mhigh
        self._sample = None
        self._mtot = None

    def sample(self, n, seed=None):
        size = n
        ran_mass = []
        np.random.seed(seed)
        while len(ran_mass) < n:
            ran_imf = np.random.uniform(0.0, self.peak, size=size)
            ran_mass = np.exp(np.random.uniform(np.log(self.mlow),
                                                np.log(self.mhigh), size=size))
            ran_mass = ran_mass[np.where((ran_imf < self.func(ran_mass)))]
            size *= int(1+n/len(ran_mass))
        self._sample = ran_mass[:n]
        self._mtot = float(np.sum(self._sample))

        plot_imf(self._sample, 0.01, 150.0, self.__name__)

        return self._sample


class IMF(object):
    """

    """
    @staticmethod
    def equalmass():
        min_mlow = 0.5
        max_mhigh = 2.0
        mlow = 1.0
        mhigh = 1.0

        def imf_func(m):
            return (1.0+m)-m

        return IMFSample(imf_func, min_mlow,
                         max_mhigh, mlow, mhigh,
                         'equalmass')

    @staticmethod
    def salpeter1955(mlow, mhigh):
        min_mlow = 0.4
        max_mhigh = 120.0

        def imf_func(m):
            return m**(-1.35)

        return IMFSample(imf_func, min_mlow,
                         max_mhigh, mlow, mhigh,
                         'salpeter1955')

    @staticmethod
    def padoan2007(mlow, mhigh):
        from scipy.special import erf
        gamma = 1.4
        m_ch = 1.0
        sigma = 1.8
        min_mlow = 0.004
        max_mhigh = 120.0

        def imf_func(m):
            return ((m**(-gamma)) * (1.0 + erf((
                4.0 * np.log(m/m_ch) + sigma**2) / (np.sqrt(8.0) * sigma))))

        return IMFSample(imf_func, min_mlow,
                         max_mhigh, mlow, mhigh,
                         'padoan2007')

    @staticmethod
    def parravano2011(mlow, mhigh):
        min_mlow = 0.004
        max_mhigh = 120.0

        def imf_func(m):
            return (m**(-1.35)) * (1.0 - np.exp(-(m / 0.35)**(0.51 + 1.35)))

        return IMFSample(imf_func, min_mlow,
                         max_mhigh, mlow, mhigh,
                         'parravano2011')

    @staticmethod
    def maschberger2013(mlow=0.01, mhigh=150.0):
        class IMFSample2(object):
            def __init__(self, mu, alpha, beta, name='maschberger2013'):
                self.mu = mu
                self.alpha = alpha
                self.beta = beta
                self.__name__ = name

            def sample(self, n, seed=None):
                np.random.seed(seed)
                i = np.arange(n)
                u_min = (i + 0) / n
                u_max = (i + 1) / n
                u = np.random.uniform(u_min, u_max, size=n)
                u = np.random.permutation(u)

                def G(m, mu, alpha, beta):
                    return (1 + (m/mu)**(1-alpha))**(1-beta)

                mu = self.mu
                alpha = self.alpha
                beta = self.beta
                Gml = G(mlow, mu, alpha, beta)
                Gmu = G(mhigh, mu, alpha, beta)
                m = ((u * (Gmu - Gml) + Gml)**(1/(1-beta)) - 1)**(1/(1-alpha))
                m *= mu

                plot_imf(m, 0.01, 150.0, self.__name__)

                return m

        return IMFSample2(0.2, 2.3, 1.4)


# -- End of File --
