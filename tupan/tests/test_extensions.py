# -*- coding: utf-8 -*-
#

"""
Test suite for extensions module.
"""


from __future__ import print_function
import unittest
import numpy as np
from pprint import pprint
from tupan.lib import gravity
from tupan.lib.utils import ctype
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
    if npart < 2:
        npart = 2
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
        print(
            "\ntest01: C(CPU) vs CL(device): max deviation of grav-phi "
            "among all combinations of i- and j-particles:",
            end=" "
        )

        npart = self.small_system.n
        deviations = []

        for i in range(1, npart+1):
            for j in range(1, npart+1):
                res = {'c': None, 'cl': None}

                # setup data
                iobj = self.small_system[:i]
                jobj = self.small_system[:j]

                # calculating on CPU
                kernel = gravity.Phi("c", ctype.prec)
                kernel.set_args(iobj, jobj)
                kernel.run()
                res['c'] = kernel.get_result()

                # calculating on GPU
                kernel = gravity.Phi("cl", ctype.prec)
                kernel.set_args(iobj, jobj)
                kernel.run()
                res['cl'] = kernel.get_result()

                # calculating deviation of result
                deviation = np.abs(res['c'] - res['cl'])
                deviations.append(deviation.max())

        deviations = np.array(deviations)
        print(deviations.max())

    def test02(self):
        print(
            "\ntest02: C(CPU) vs CL(device): max deviation of grav-acc "
            "among all combinations of i- and j-particles:",
            end=" "
        )

        npart = self.small_system.n
        deviations = []

        for i in range(1, npart+1):
            for j in range(1, npart+1):
                res = {'c': None, 'cl': None}

                # setup data
                iobj = self.small_system[:i]
                jobj = self.small_system[:j]

                # calculating on CPU
                kernel = gravity.Acc("c", ctype.prec)
                kernel.set_args(iobj, jobj)
                kernel.run()
                res['c'] = kernel.get_result()

                # calculating on GPU
                kernel = gravity.Acc("cl", ctype.prec)
                kernel.set_args(iobj, jobj)
                kernel.run()
                res['cl'] = kernel.get_result()

                # calculating deviation of result
                deviation = np.sqrt(
                    sum((a_c-a_cl)**2
                        for a_c, a_cl in zip(res['c'], res['cl']))
                )
                deviations.append(deviation.max())

        deviations = np.array(deviations)
        print(deviations.max())

    def test03(self):
        print(
            "\ntest03: C(CPU) vs CL(device): max deviation of grav-acc_jerk "
            "among all combinations of i- and j-particles:",
            end=" "
        )

        npart = self.small_system.n
        deviations = []

        for i in range(1, npart+1):
            for j in range(1, npart+1):
                res = {'c': None, 'cl': None}

                # setup data
                iobj = self.small_system[:i]
                jobj = self.small_system[:j]

                # calculating on CPU
                kernel = gravity.AccJerk("c", ctype.prec)
                kernel.set_args(iobj, jobj)
                kernel.run()
                res['c'] = kernel.get_result()

                # calculating on GPU
                kernel = gravity.AccJerk("cl", ctype.prec)
                kernel.set_args(iobj, jobj)
                kernel.run()
                res['cl'] = kernel.get_result()

                # calculating deviation of result
                deviation = np.sqrt(
                    sum((aj_c-aj_cl)**2
                        for aj_c, aj_cl in zip(res['c'], res['cl']))
                )
                deviations.append(deviation.max())

        deviations = np.array(deviations)
        print(deviations.max())

    def test04(self):
        print(
            "\ntest04: C(CPU) vs CL(device): max deviation of grav-tstep "
            "among all combinations of i- and j-particles:",
            end=" "
        )

        eta = 1.0/64
        npart = self.small_system.n
        deviations = []

        for i in range(1, npart+1):
            for j in range(1, npart+1):
                res = {'c': None, 'cl': None}

                # setup data
                iobj = self.small_system[:i]
                jobj = self.small_system[:j]

                # calculating on CPU
                kernel = gravity.Tstep("c", ctype.prec)
                kernel.set_args(iobj, jobj, eta)
                kernel.run()
                res['c'] = kernel.get_result()

                # calculating on GPU
                kernel = gravity.Tstep("cl", ctype.prec)
                kernel.set_args(iobj, jobj, eta)
                kernel.run()
                res['cl'] = kernel.get_result()

                # calculating deviation of result
                deviation0 = np.abs(res['c'][0] - res['cl'][0])
                deviation1 = np.abs(res['c'][1] - res['cl'][1])
                deviations.append([deviation0.max(), deviation1.max()])

        deviations = np.array(deviations)
        print(deviations[:, 0].max(), deviations[:, 1].max())

    def test05(self):
        print(
            "\ntest05: C(CPU) vs CL(device): max deviation of grav-pnacc "
            "among all combinations of i- and j-particles:",
            end=" "
        )

        npart = self.small_system.n
        deviations = []

        gravity.clight.pn_order = 7
        gravity.clight.clight = 128

        for i in range(1, npart+1):
            for j in range(1, npart+1):
                res = {'c': None, 'cl': None}

                # setup data
                iobj = self.small_system[:i]
                jobj = self.small_system[:j]

                # calculating on CPU
                kernel = gravity.PNAcc("c", ctype.prec)
                kernel.set_args(iobj, jobj)
                kernel.run()
                res['c'] = kernel.get_result()

                # calculating on GPU
                kernel = gravity.PNAcc("cl", ctype.prec)
                kernel.set_args(iobj, jobj)
                kernel.run()
                res['cl'] = kernel.get_result()

                # calculating deviation of result
                deviation = np.sqrt(sum((
                    a_c-a_cl)**2 for a_c, a_cl in zip(res['c'], res['cl'])))
                deviations.append(deviation.max())

        deviations = np.array(deviations)
        print(deviations.max())

    def test06(self):
        print(
            "\ntest06: C(CPU) vs CL(device): performance of grav-phi:",
            end=" "
        )

        n = 5   # no. of samples
        best = {'set_args': {'c': None, 'cl': None},
                'run': {'c': None, 'cl': None},
                'get_result': {'c': None, 'cl': None}}

        # setup data
        iobj = self.large_system
        jobj = self.large_system

        # calculating using SP on CPU
        kernel = gravity.Phi("c", ctype.prec)
        best['set_args']['c'] = best_of(n, kernel.set_args, iobj, jobj)
        best['run']['c'] = best_of(n, kernel.run)
        best['get_result']['c'] = best_of(n, kernel.get_result)

        # calculating using DP on CPU
        kernel = gravity.Phi("cl", ctype.prec)
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

    def test07(self):
        print(
            "\ntest07: C(CPU) vs CL(device): performance of grav-acc:",
            end=" "
        )

        n = 5   # no. of samples
        best = {'set_args': {'c': None, 'cl': None},
                'run': {'c': None, 'cl': None},
                'get_result': {'c': None, 'cl': None}}

        # setup data
        iobj = self.large_system
        jobj = self.large_system

        # calculating using SP on CPU
        kernel = gravity.Acc("c", ctype.prec)
        best['set_args']['c'] = best_of(n, kernel.set_args, iobj, jobj)
        best['run']['c'] = best_of(n, kernel.run)
        best['get_result']['c'] = best_of(n, kernel.get_result)

        # calculating using DP on CPU
        kernel = gravity.Acc("cl", ctype.prec)
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

    def test08(self):
        print(
            "\ntest08: C(CPU) vs CL(device): performance of grav-pnacc:",
            end=" "
        )

        n = 5   # no. of samples
        best = {'set_args': {'c': None, 'cl': None},
                'run': {'c': None, 'cl': None},
                'get_result': {'c': None, 'cl': None}}

        # setup data
        iobj = self.large_system
        jobj = self.large_system

        gravity.clight.pn_order = 7
        gravity.clight.clight = 128

        # calculating using SP on CPU
        kernel = gravity.PNAcc("c", ctype.prec)
        best['set_args']['c'] = best_of(n, kernel.set_args, iobj, jobj)
        best['run']['c'] = best_of(n, kernel.run)
        best['get_result']['c'] = best_of(n, kernel.get_result)

        # calculating using DP on CPU
        kernel = gravity.PNAcc("cl", ctype.prec)
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


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCase)
    unittest.TextTestRunner(verbosity=1).run(suite)


########## end of file ##########
