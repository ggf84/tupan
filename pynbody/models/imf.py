#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
from scipy import (integrate, optimize)
import numpy as np
import random
import math



class IMFSample(object):
    """

    """
    def __new__(cls, *args):
        return object.__new__(cls)


    def __init__(self, imf_func, min_mlow, max_mhigh, mlow, mhigh):
        """

        """
        if mlow < min_mlow or mhigh > max_mhigh:
            raise ValueError('Mass range outer of limits'
                             ' [{0}:{1}].'.format(min_mlow, max_mhigh))

        intarg = lambda m: imf_func(m)/m
        (norm, err) = integrate.quad(intarg, min_mlow, max_mhigh)
        imf_func_normed = lambda m: imf_func(m)/norm
        mpeak = optimize.fminbound(lambda m: -imf_func_normed(m),
                                   min_mlow, max_mhigh,
                                   xtol=1.0e-8).item()
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


    def sample(self, n=0):
        if n:
            msample = []
            for i in range(n):
                ran_imf = self.peak
                ran_mass = self.mhigh
                while (ran_imf > self.func(ran_mass)):
                    ran_imf = random.uniform(0.0, self.peak)
                    ran_mass = math.exp(random.uniform(math.log(self.mlow),
                                                       math.log(self.mhigh)))
                msample.append(ran_mass)
            self._sample = np.array(msample, dtype=np.float64)
            self._mtot = np.sum(self._sample)
        return self._sample




class IMF(object):
    """

    """
    def __new__(cls):
        return object.__new__(cls)


    @classmethod
    def equal(self):
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
        min_mlow = 0.5
        max_mhigh = 120.0
        imf = IMFSample(imf_func, min_mlow, max_mhigh, mlow, mhigh)
        return imf


    @classmethod
    def padoan2007(self, mlow, mhigh):
        from scipy.special import erf
        Gamma = 1.4
        m_ch = 1.0
        sigma = 1.8
        imf_func = lambda m: ((m**(-Gamma)) * (1.0 + erf((4.0 * np.log(m/m_ch) + sigma**2) / (np.sqrt(8.0) * sigma))))
        min_mlow = 0.004
        max_mhigh = 120.0
        imf = IMFSample(imf_func, min_mlow, max_mhigh, mlow, mhigh)
        return imf


    @classmethod
    def parravano2011(self, mlow, mhigh):
        imf_func = lambda m: (m**(-1.35))*(1.0 - np.exp(-(m/0.35)**(0.51+1.35)))
        min_mlow = 0.004
        max_mhigh = 120.0
        imf = IMFSample(imf_func, min_mlow, max_mhigh, mlow, mhigh)
        return imf





########## end of file ##########
