# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function
import numpy as np
from scipy.integrate import quad
from scipy.optimize import fminbound
from ..lib.utils.timing import decallmethods, timings


__all__ = ['IMF']


@decallmethods(timings)
class IMFSample(object):
    """

    """
    def __init__(self, imf_func, min_mlow, max_mhigh, mlow, mhigh):
        """

        """
        if mlow < min_mlow or mhigh > max_mhigh:
            raise ValueError('Mass range outer of limits'
                             ' [{0}:{1}].'.format(min_mlow, max_mhigh))

        intarg = lambda m: imf_func(m)/m
        (norm, err) = quad(intarg, min_mlow, max_mhigh)
        imf_func_normed = lambda m: imf_func(m)/norm
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

    def sample(self, n):
        size = n
        ran_mass = []
        while len(ran_mass) < n:
            ran_imf = np.random.uniform(0.0, self.peak, size=size)
            ran_mass = np.exp(np.random.uniform(np.log(self.mlow),
                                                np.log(self.mhigh), size=size))
            ran_mass = ran_mass[np.where((ran_imf < self.func(ran_mass)))]
            size *= int(1+n/len(ran_mass))
        self._sample = ran_mass[:n]
        self._mtot = float(np.sum(self._sample))
        return self._sample


class IMF(object):
    """

    """
    @classmethod
    def equalmass(self):
        imf_func = lambda m: (1.0+m)-m
        min_mlow = 0.1
        max_mhigh = 10.0
        mlow = 1.0
        mhigh = 1.0
        imf = IMFSample(imf_func, min_mlow, max_mhigh, mlow, mhigh)
        return imf

    @classmethod
    def salpeter1955(self, mlow, mhigh):
        imf_func = lambda m: m**(-1.35)
        min_mlow = 0.4
        max_mhigh = 120.0
        imf = IMFSample(imf_func, min_mlow, max_mhigh, mlow, mhigh)
        return imf

    @classmethod
    def padoan2007(self, mlow, mhigh):
        from scipy.special import erf
        Gamma = 1.4
        m_ch = 1.0
        sigma = 1.8
        imf_func = lambda m: ((m**(-Gamma)) * (1.0 + erf((
            4.0 * np.log(m/m_ch) + sigma**2) / (np.sqrt(8.0) * sigma))))
        min_mlow = 0.004
        max_mhigh = 120.0
        imf = IMFSample(imf_func, min_mlow, max_mhigh, mlow, mhigh)
        return imf

    @classmethod
    def parravano2011(self, mlow, mhigh):
        imf_func = lambda m: (m**(-1.35))*(
            1.0 - np.exp(-(m/0.35)**(0.51+1.35)))
        min_mlow = 0.004
        max_mhigh = 120.0
        imf = IMFSample(imf_func, min_mlow, max_mhigh, mlow, mhigh)
        return imf


# -- End of File --
