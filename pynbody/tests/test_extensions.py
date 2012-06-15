#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for extensions module.
"""


from __future__ import print_function
from pprint import pprint
import unittest
import numpy as np
from pynbody.lib import gravity
from pynbody.lib.extensions import libkernels
from pynbody.lib.utils.timing import Timer


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
    from pynbody.ics.imf import IMF
    from pynbody.ics.plummer import Plummer
    imf = IMF.padoan2007(0.075, 120.0)
    p = Plummer(npart, imf, eps=0.0, seed=1)
    p.make_plummer()
    bi = p.particles['body']
    return bi

small_system = set_particles(32)
large_system = set_particles(4096)



class TestCase(unittest.TestCase):

    def test01(self):
        print('\ntest01: max deviation of grav-phi (in SP on CPU and GPU) between all combinations of i- and j-particles:', end=' ')

        npart = small_system.n
        deviations = []

        for i in range(1, npart+1):
            for j in range(1, npart+1):
                result = {'cpu': None, 'gpu': None}

                # setup data
                iobj = small_system[:i].copy()
                jobj = small_system[:j].copy()

                # calculating on CPU
                kernel = gravity.Phi(libkernels['sp']['c'])
                kernel.set_args(iobj, jobj)
                kernel.run()
                result['cpu'] = kernel.get_result()

                # calculating on GPU
                kernel = gravity.Phi(libkernels['sp']['cl'])
                kernel.set_args(iobj, jobj)
                kernel.run()
                result['gpu'] = kernel.get_result()

                # calculating deviation of result
                deviation = np.abs(result['cpu'] - result['gpu'])
                deviations.append(deviation.max())

        deviations = np.array(deviations)
        print(deviations.max())


    def test02(self):
        print('\ntest02: max deviation of grav-acc (in SP on CPU and GPU) between all combinations of i- and j-particles:', end=' ')

        npart = small_system.n
        deviations = []

        for i in range(1, npart+1):
            for j in range(1, npart+1):
                result = {'cpu': None, 'gpu': None}

                # setup data
                iobj = small_system[:i].copy()
                jobj = small_system[:j].copy()

                # calculating on CPU
                kernel = gravity.Acc(libkernels['sp']['c'])
                kernel.set_args(iobj, jobj)
                kernel.run()
                result['cpu'] = kernel.get_result()

                # calculating on GPU
                kernel = gravity.Acc(libkernels['sp']['cl'])
                kernel.set_args(iobj, jobj)
                kernel.run()
                result['gpu'] = kernel.get_result()

                # calculating deviation of result
                deviation = np.sqrt(((result['cpu']-result['gpu'])**2).sum(1))
                deviations.append(deviation.max())

        deviations = np.array(deviations)
        print(deviations.max())


    def test03(self):
        print('\ntest03: max deviation of grav-pnacc (in SP on CPU and GPU) between all combinations of i- and j-particles:', end=' ')

        npart = small_system.n
        deviations = []

        for i in range(1, npart+1):
            for j in range(1, npart+1):
                result = {'cpu': None, 'gpu': None}

                # setup data
                iobj = small_system[:i].copy()
                jobj = small_system[:j].copy()

                # calculating on CPU
                kernel = gravity.PNAcc(libkernels['sp']['c'])
                kernel.set_args(iobj, jobj, 7, 128)
                kernel.run()
                result['cpu'] = kernel.get_result()

                # calculating on GPU
                kernel = gravity.PNAcc(libkernels['sp']['cl'])
                kernel.set_args(iobj, jobj, 7, 128)
                kernel.run()
                result['gpu'] = kernel.get_result()

                # calculating deviation of result
                deviation = np.sqrt(((result['cpu']-result['gpu'])**2).sum(1))
                deviations.append(deviation.max())

        deviations = np.array(deviations)
        print(deviations.max())


    def test04(self):
        print('\ntest04: max deviation of grav-phi (in DP on CPU and GPU) between all combinations of i- and j-particles:', end=' ')

        npart = small_system.n
        deviations = []

        for i in range(1, npart+1):
            for j in range(1, npart+1):
                result = {'cpu': None, 'gpu': None}

                # setup data
                iobj = small_system[:i].copy()
                jobj = small_system[:j].copy()

                # calculating on CPU
                kernel = gravity.Phi(libkernels['dp']['c'])
                kernel.set_args(iobj, jobj)
                kernel.run()
                result['cpu'] = kernel.get_result()

                # calculating on GPU
                kernel = gravity.Phi(libkernels['dp']['cl'])
                kernel.set_args(iobj, jobj)
                kernel.run()
                result['gpu'] = kernel.get_result()

                # calculating deviation of result
                deviation = np.abs(result['cpu'] - result['gpu'])
                deviations.append(deviation.max())

        deviations = np.array(deviations)
        print(deviations.max())


    def test05(self):
        print('\ntest05: max deviation of grav-acc (in DP on CPU and GPU) between all combinations of i- and j-particles:', end=' ')

        npart = small_system.n
        deviations = []

        for i in range(1, npart+1):
            for j in range(1, npart+1):
                result = {'cpu': None, 'gpu': None}

                # setup data
                iobj = small_system[:i].copy()
                jobj = small_system[:j].copy()

                # calculating on CPU
                kernel = gravity.Acc(libkernels['dp']['c'])
                kernel.set_args(iobj, jobj)
                kernel.run()
                result['cpu'] = kernel.get_result()

                # calculating on GPU
                kernel = gravity.Acc(libkernels['dp']['cl'])
                kernel.set_args(iobj, jobj)
                kernel.run()
                result['gpu'] = kernel.get_result()

                # calculating deviation of result
                deviation = np.sqrt(((result['cpu']-result['gpu'])**2).sum(1))
                deviations.append(deviation.max())

        deviations = np.array(deviations)
        print(deviations.max())


    def test06(self):
        print('\ntest06: max deviation of grav-pnacc (in DP on CPU and GPU) between all combinations of i- and j-particles:', end=' ')

        npart = small_system.n
        deviations = []

        for i in range(1, npart+1):
            for j in range(1, npart+1):
                result = {'cpu': None, 'gpu': None}

                # setup data
                iobj = small_system[:i].copy()
                jobj = small_system[:j].copy()

                # calculating on CPU
                kernel = gravity.PNAcc(libkernels['dp']['c'])
                kernel.set_args(iobj, jobj, 7, 128)
                kernel.run()
                result['cpu'] = kernel.get_result()

                # calculating on GPU
                kernel = gravity.PNAcc(libkernels['dp']['cl'])
                kernel.set_args(iobj, jobj, 7, 128)
                kernel.run()
                result['gpu'] = kernel.get_result()

                # calculating deviation of result
                deviation = np.sqrt(((result['cpu']-result['gpu'])**2).sum(1))
                deviations.append(deviation.max())

        deviations = np.array(deviations)
        print(deviations.max())


    def test07(self):
        print('\ntest07: performance of grav-phi (in SP and DP on CPU):', end=' ')

        n = 3   # no. of samples
        best = {'set_args': {'sp': None, 'dp': None},
                'run': {'sp': None, 'dp': None},
                'get_result': {'sp': None, 'dp': None}}

        # setup data
        iobj = large_system.copy()
        jobj = large_system.copy()

        # calculating using SP on CPU
        kernel = gravity.Phi(libkernels['sp']['c'])
        best['set_args']['sp'] = best_of(n, kernel.set_args, iobj, jobj)
        best['run']['sp'] = best_of(n, kernel.run)
        best['get_result']['sp'] = best_of(n, kernel.get_result)

        # calculating using DP on CPU
        kernel = gravity.Phi(libkernels['dp']['c'])
        best['set_args']['dp'] = best_of(n, kernel.set_args, iobj, jobj)
        best['run']['dp'] = best_of(n, kernel.run)
        best['get_result']['dp'] = best_of(n, kernel.get_result)

        total = {'sp': 0.0, 'dp': 0.0}
        for d in best.values():
            for k, v in d.items():
                total[k] += v
        for key in best.keys():
            for k, v in total.items():
                best[key][k] /= v / 100
        print('total time:', total)
        print('percentage:')
        pprint(best, width=40)


    def test08(self):
        print('\ntest08: performance of grav-acc (in SP and DP on CPU):', end=' ')

        n = 3   # no. of samples
        best = {'set_args': {'sp': None, 'dp': None},
                'run': {'sp': None, 'dp': None},
                'get_result': {'sp': None, 'dp': None}}

        # setup data
        iobj = large_system.copy()
        jobj = large_system.copy()

        # calculating using SP on CPU
        kernel = gravity.Acc(libkernels['sp']['c'])
        best['set_args']['sp'] = best_of(n, kernel.set_args, iobj, jobj)
        best['run']['sp'] = best_of(n, kernel.run)
        best['get_result']['sp'] = best_of(n, kernel.get_result)

        # calculating using DP on CPU
        kernel = gravity.Acc(libkernels['dp']['c'])
        best['set_args']['dp'] = best_of(n, kernel.set_args, iobj, jobj)
        best['run']['dp'] = best_of(n, kernel.run)
        best['get_result']['dp'] = best_of(n, kernel.get_result)

        total = {'sp': 0.0, 'dp': 0.0}
        for d in best.values():
            for k, v in d.items():
                total[k] += v
        for key in best.keys():
            for k, v in total.items():
                best[key][k] /= v / 100
        print('total time:', total)
        print('percentage:')
        pprint(best, width=40)


    def test09(self):
        print('\ntest09: performance of grav-pnacc (in SP and DP on CPU):', end=' ')

        n = 3   # no. of samples
        best = {'set_args': {'sp': None, 'dp': None},
                'run': {'sp': None, 'dp': None},
                'get_result': {'sp': None, 'dp': None}}

        # setup data
        iobj = large_system.copy()
        jobj = large_system.copy()

        # calculating using SP on CPU
        kernel = gravity.PNAcc(libkernels['sp']['c'])
        best['set_args']['sp'] = best_of(n, kernel.set_args, iobj, jobj, 7, 128)
        best['run']['sp'] = best_of(n, kernel.run)
        best['get_result']['sp'] = best_of(n, kernel.get_result)

        # calculating using DP on CPU
        kernel = gravity.PNAcc(libkernels['dp']['c'])
        best['set_args']['dp'] = best_of(n, kernel.set_args, iobj, jobj, 7, 128)
        best['run']['dp'] = best_of(n, kernel.run)
        best['get_result']['dp'] = best_of(n, kernel.get_result)

        total = {'sp': 0.0, 'dp': 0.0}
        for d in best.values():
            for k, v in d.items():
                total[k] += v
        for key in best.keys():
            for k, v in total.items():
                best[key][k] /= v / 100
        print('total time:', total)
        print('percentage:')
        pprint(best, width=40)


    def test10(self):
        print('\ntest10: performance of grav-phi (in SP and DP on GPU):', end=' ')

        n = 3   # no. of samples
        best = {'set_args': {'sp': None, 'dp': None},
                'run': {'sp': None, 'dp': None},
                'get_result': {'sp': None, 'dp': None}}

        # setup data
        iobj = large_system.copy()
        jobj = large_system.copy()

        # calculating using SP on CPU
        kernel = gravity.Phi(libkernels['sp']['cl'])
        best['set_args']['sp'] = best_of(n, kernel.set_args, iobj, jobj)
        best['run']['sp'] = best_of(n, kernel.run)
        best['get_result']['sp'] = best_of(n, kernel.get_result)

        # calculating using DP on CPU
        kernel = gravity.Phi(libkernels['dp']['cl'])
        best['set_args']['dp'] = best_of(n, kernel.set_args, iobj, jobj)
        best['run']['dp'] = best_of(n, kernel.run)
        best['get_result']['dp'] = best_of(n, kernel.get_result)

        total = {'sp': 0.0, 'dp': 0.0}
        for d in best.values():
            for k, v in d.items():
                total[k] += v
        for key in best.keys():
            for k, v in total.items():
                best[key][k] /= v / 100
        print('total time:', total)
        print('percentage:')
        pprint(best, width=40)


    def test11(self):
        print('\ntest11: performance of grav-acc (in SP and DP on GPU):', end=' ')

        n = 3   # no. of samples
        best = {'set_args': {'sp': None, 'dp': None},
                'run': {'sp': None, 'dp': None},
                'get_result': {'sp': None, 'dp': None}}

        # setup data
        iobj = large_system.copy()
        jobj = large_system.copy()

        # calculating using SP on CPU
        kernel = gravity.Acc(libkernels['sp']['cl'])
        best['set_args']['sp'] = best_of(n, kernel.set_args, iobj, jobj)
        best['run']['sp'] = best_of(n, kernel.run)
        best['get_result']['sp'] = best_of(n, kernel.get_result)

        # calculating using DP on CPU
        kernel = gravity.Acc(libkernels['dp']['cl'])
        best['set_args']['dp'] = best_of(n, kernel.set_args, iobj, jobj)
        best['run']['dp'] = best_of(n, kernel.run)
        best['get_result']['dp'] = best_of(n, kernel.get_result)

        total = {'sp': 0.0, 'dp': 0.0}
        for d in best.values():
            for k, v in d.items():
                total[k] += v
        for key in best.keys():
            for k, v in total.items():
                best[key][k] /= v / 100
        print('total time:', total)
        print('percentage:')
        pprint(best, width=40)


    def test12(self):
        print('\ntest12: performance of grav-pnacc (in SP and DP on GPU):', end=' ')

        n = 3   # no. of samples
        best = {'set_args': {'sp': None, 'dp': None},
                'run': {'sp': None, 'dp': None},
                'get_result': {'sp': None, 'dp': None}}

        # setup data
        iobj = large_system.copy()
        jobj = large_system.copy()

        # calculating using SP on CPU
        kernel = gravity.PNAcc(libkernels['sp']['cl'])
        best['set_args']['sp'] = best_of(n, kernel.set_args, iobj, jobj, 7, 128)
        best['run']['sp'] = best_of(n, kernel.run)
        best['get_result']['sp'] = best_of(n, kernel.get_result)

        # calculating using DP on CPU
        kernel = gravity.PNAcc(libkernels['dp']['cl'])
        best['set_args']['dp'] = best_of(n, kernel.set_args, iobj, jobj, 7, 128)
        best['run']['dp'] = best_of(n, kernel.run)
        best['get_result']['dp'] = best_of(n, kernel.get_result)

        total = {'sp': 0.0, 'dp': 0.0}
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
    unittest.main()


########## end of file ##########
