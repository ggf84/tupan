#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for extensions module.
"""


from __future__ import print_function
from pprint import pprint
import unittest
import numpy as np
from tupan.lib import gravity
from tupan.lib.extensions import libkernels
from tupan.lib.utils.timing import Timer


def best_of(n, func, *args, **kwargs):
    elapsed = []
    for i in range(n):
        timer = Timer()
        timer.start()
        ret = func(*args, **kwargs)
    elapsed.append(timer.elapsed())
    return min(elapsed)


def set_particles(npart):
    if npart < 2: npart = 2
    from tupan.ics.imf import IMF
    from tupan.ics.plummer import Plummer
    imf = IMF.padoan2007(0.075, 120.0)
    p = Plummer(npart, imf, eps=0.0, seed=1)
    p.make_plummer()
    return p.particles




class TestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.small_system = set_particles(16)
        cls.large_system = set_particles(1024)


    def test01(self):
        print('\ntest01: C(CPU) vs CL(device): max deviation of grav-phi among all combinations of i- and j-particles:', end=' ')

        npart = self.small_system.n
        deviations = []

        for i in range(1, npart+1):
            for j in range(1, npart+1):
                res = {'c': None, 'cl': None}

                # setup data
                iobj = self.small_system[:i]
                jobj = self.small_system[:j]

                # calculating on CPU
                kernel = gravity.Phi(libkernels['c'])
                kernel.set_args(iobj, jobj)
                kernel.run()
                res['c'] = kernel.get_result()

                # calculating on GPU
                kernel = gravity.Phi(libkernels['cl'])
                kernel.set_args(iobj, jobj)
                kernel.run()
                res['cl'] = kernel.get_result()

                # calculating deviation of result
                deviation = np.abs(res['c'] - res['cl'])
                deviations.append(deviation.max())

        deviations = np.array(deviations)
        print(deviations.max())


    def test02(self):
        print('\ntest02: C(CPU) vs CL(device): max deviation of grav-acc among all combinations of i- and j-particles:', end=' ')

        npart = self.small_system.n
        deviations = []

        for i in range(1, npart+1):
            for j in range(1, npart+1):
                res = {'c': None, 'cl': None}

                # setup data
                iobj = self.small_system[:i]
                jobj = self.small_system[:j]

                # calculating on CPU
                kernel = gravity.Acc(libkernels['c'])
                kernel.set_args(iobj, jobj)
                kernel.run()
                res['c'] = kernel.get_result()

                # calculating on GPU
                kernel = gravity.Acc(libkernels['cl'])
                kernel.set_args(iobj, jobj)
                kernel.run()
                res['cl'] = kernel.get_result()

                # calculating deviation of result
                deviation = np.sqrt(sum((a_c-a_cl)**2 for a_c, a_cl in zip(res['c'], res['cl'])))
                deviations.append(deviation.max())

        deviations = np.array(deviations)
        print(deviations.max())


    def test03(self):
        print('\ntest03: C(CPU) vs CL(device): max deviation of grav-pnacc among all combinations of i- and j-particles:', end=' ')

        npart = self.small_system.n
        deviations = []

        for i in range(1, npart+1):
            for j in range(1, npart+1):
                res = {'c': None, 'cl': None}

                # setup data
                iobj = self.small_system[:i]
                jobj = self.small_system[:j]

                # calculating on CPU
                kernel = gravity.PNAcc(libkernels['c'])
                kernel.set_args(iobj, jobj, 7, 128)
                kernel.run()
                res['c'] = kernel.get_result()

                # calculating on GPU
                kernel = gravity.PNAcc(libkernels['cl'])
                kernel.set_args(iobj, jobj, 7, 128)
                kernel.run()
                res['cl'] = kernel.get_result()

                # calculating deviation of result
                deviation = np.sqrt(sum((a_c-a_cl)**2 for a_c, a_cl in zip(res['c'], res['cl'])))
                deviations.append(deviation.max())

        deviations = np.array(deviations)
        print(deviations.max())


    def test04(self):
        print('\ntest04: C(CPU) vs CL(device): performance of grav-phi:', end=' ')

        n = 5   # no. of samples
        best = {'set_args': {'c': None, 'cl': None},
                'run': {'c': None, 'cl': None},
                'get_result': {'c': None, 'cl': None}}

        # setup data
        iobj = self.large_system
        jobj = self.large_system

        # calculating using SP on CPU
        kernel = gravity.Phi(libkernels['c'])
        best['set_args']['c'] = best_of(n, kernel.set_args, iobj, jobj)
        best['run']['c'] = best_of(n, kernel.run)
        best['get_result']['c'] = best_of(n, kernel.get_result)

        # calculating using DP on CPU
        kernel = gravity.Phi(libkernels['cl'])
        best['set_args']['cl'] = best_of(n, kernel.set_args, iobj, jobj)
        best['run']['cl'] = best_of(n, kernel.run)
        best['get_result']['cl'] = best_of(n, kernel.get_result)

        total = {'c': 0.0, 'cl': 0.0}
        for d in best.values():
            for k, v in d.items():
                total[k] += v
        for key in best.keys():
            for k, v in total.items():
                best[key][k] /= v / 100
        print('total time:', total)
        print('percentage:')
        pprint(best, width=40)


    def test05(self):
        print('\ntest05: C(CPU) vs CL(device): performance of grav-acc:', end=' ')

        n = 5   # no. of samples
        best = {'set_args': {'c': None, 'cl': None},
                'run': {'c': None, 'cl': None},
                'get_result': {'c': None, 'cl': None}}

        # setup data
        iobj = self.large_system
        jobj = self.large_system

        # calculating using SP on CPU
        kernel = gravity.Acc(libkernels['c'])
        best['set_args']['c'] = best_of(n, kernel.set_args, iobj, jobj)
        best['run']['c'] = best_of(n, kernel.run)
        best['get_result']['c'] = best_of(n, kernel.get_result)

        # calculating using DP on CPU
        kernel = gravity.Acc(libkernels['cl'])
        best['set_args']['cl'] = best_of(n, kernel.set_args, iobj, jobj)
        best['run']['cl'] = best_of(n, kernel.run)
        best['get_result']['cl'] = best_of(n, kernel.get_result)

        total = {'c': 0.0, 'cl': 0.0}
        for d in best.values():
            for k, v in d.items():
                total[k] += v
        for key in best.keys():
            for k, v in total.items():
                best[key][k] /= v / 100
        print('total time:', total)
        print('percentage:')
        pprint(best, width=40)


    def test06(self):
        print('\ntest06: C(CPU) vs CL(device): performance of grav-pnacc:', end=' ')

        n = 5   # no. of samples
        best = {'set_args': {'c': None, 'cl': None},
                'run': {'c': None, 'cl': None},
                'get_result': {'c': None, 'cl': None}}

        # setup data
        iobj = self.large_system
        jobj = self.large_system

        # calculating using SP on CPU
        kernel = gravity.PNAcc(libkernels['c'])
        best['set_args']['c'] = best_of(n, kernel.set_args, iobj, jobj, 7, 128)
        best['run']['c'] = best_of(n, kernel.run)
        best['get_result']['c'] = best_of(n, kernel.get_result)

        # calculating using DP on CPU
        kernel = gravity.PNAcc(libkernels['cl'])
        best['set_args']['cl'] = best_of(n, kernel.set_args, iobj, jobj, 7, 128)
        best['run']['cl'] = best_of(n, kernel.run)
        best['get_result']['cl'] = best_of(n, kernel.get_result)

        total = {'c': 0.0, 'cl': 0.0}
        for d in best.values():
            for k, v in d.items():
                total[k] += v
        for key in best.keys():
            for k, v in total.items():
                best[key][k] /= v / 100
        print('total time:', total)
        print('percentage:')
        pprint(best, width=40)



if __name__ == "__main__":
    import sys
    if "--use_sp" in sys.argv: sys.argv.remove("--use_sp")
    unittest.main()


########## end of file ##########
