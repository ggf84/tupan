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
    return p.ps


class TestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.small_system = set_particles(41)
        cls.large_system = set_particles(1151)

    def test01(self):
        print(
            "\ntest01: C(CPU) vs CL(device): max deviation of grav-phi "
            "among all combinations of i- and j-particles:",
            end=" "
        )

        npart = self.small_system.n
        deviations = []

        kernel_c = gravity.Phi("c", ctype.prec)
        kernel_cl = gravity.Phi("cl", ctype.prec)

        iobj = self.small_system
        for j in range(1, npart+1):
            res = {'c': None, 'cl': None}

            # setup data
            jobj = self.small_system[:j]

            # calculating on CPU
            kernel_c.set_args(iobj, jobj)
            kernel_c.run()
            res['c'] = kernel_c.get_result()

            # calculating on GPU
            kernel_cl.set_args(iobj, jobj)
            kernel_cl.run()
            res['cl'] = kernel_cl.get_result()

            # calculating deviation of result
            deviation = np.sqrt(
                sum((c_res-cl_res)**2
                    for c_res, cl_res in zip(res['c'], res['cl']))
            )
            deviations.append(deviation.max())

        jobj = self.small_system
        for i in range(1, npart+1):
            res = {'c': None, 'cl': None}

            # setup data
            iobj = self.small_system[:i]

            # calculating on CPU
            kernel_c.set_args(iobj, jobj)
            kernel_c.run()
            res['c'] = kernel_c.get_result()

            # calculating on GPU
            kernel_cl.set_args(iobj, jobj)
            kernel_cl.run()
            res['cl'] = kernel_cl.get_result()

            # calculating deviation of result
            deviation = np.sqrt(
                sum((c_res-cl_res)**2
                    for c_res, cl_res in zip(res['c'], res['cl']))
            )
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

        kernel_c = gravity.Acc("c", ctype.prec)
        kernel_cl = gravity.Acc("cl", ctype.prec)

        iobj = self.small_system
        for j in range(1, npart+1):
            res = {'c': None, 'cl': None}

            # setup data
            jobj = self.small_system[:j]

            # calculating on CPU
            kernel_c.set_args(iobj, jobj)
            kernel_c.run()
            res['c'] = kernel_c.get_result()

            # calculating on GPU
            kernel_cl.set_args(iobj, jobj)
            kernel_cl.run()
            res['cl'] = kernel_cl.get_result()

            # calculating deviation of result
            deviation = np.sqrt(
                sum((c_res-cl_res)**2
                    for c_res, cl_res in zip(res['c'], res['cl']))
            )
            deviations.append(deviation.max())

        jobj = self.small_system
        for i in range(1, npart+1):
            res = {'c': None, 'cl': None}

            # setup data
            iobj = self.small_system[:i]

            # calculating on CPU
            kernel_c.set_args(iobj, jobj)
            kernel_c.run()
            res['c'] = kernel_c.get_result()

            # calculating on GPU
            kernel_cl.set_args(iobj, jobj)
            kernel_cl.run()
            res['cl'] = kernel_cl.get_result()

            # calculating deviation of result
            deviation = np.sqrt(
                sum((c_res-cl_res)**2
                    for c_res, cl_res in zip(res['c'], res['cl']))
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

        kernel_c = gravity.AccJerk("c", ctype.prec)
        kernel_cl = gravity.AccJerk("cl", ctype.prec)

        iobj = self.small_system
        for j in range(1, npart+1):
            res = {'c': None, 'cl': None}

            # setup data
            jobj = self.small_system[:j]

            # calculating on CPU
            kernel_c.set_args(iobj, jobj)
            kernel_c.run()
            res['c'] = kernel_c.get_result()

            # calculating on GPU
            kernel_cl.set_args(iobj, jobj)
            kernel_cl.run()
            res['cl'] = kernel_cl.get_result()

            # calculating deviation of result
            deviation = np.sqrt(
                sum((c_res-cl_res)**2
                    for c_res, cl_res in zip(res['c'], res['cl']))
            )
            deviations.append(deviation.max())

        jobj = self.small_system
        for i in range(1, npart+1):
            res = {'c': None, 'cl': None}

            # setup data
            iobj = self.small_system[:i]

            # calculating on CPU
            kernel_c.set_args(iobj, jobj)
            kernel_c.run()
            res['c'] = kernel_c.get_result()

            # calculating on GPU
            kernel_cl.set_args(iobj, jobj)
            kernel_cl.run()
            res['cl'] = kernel_cl.get_result()

            # calculating deviation of result
            deviation = np.sqrt(
                sum((c_res-cl_res)**2
                    for c_res, cl_res in zip(res['c'], res['cl']))
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

        kernel_c = gravity.Tstep("c", ctype.prec)
        kernel_cl = gravity.Tstep("cl", ctype.prec)

        iobj = self.small_system
        for j in range(1, npart+1):
            res = {'c': None, 'cl': None}

            # setup data
            jobj = self.small_system[:j]

            # calculating on CPU
            kernel_c.set_args(iobj, jobj, eta)
            kernel_c.run()
            res['c'] = kernel_c.get_result()

            # calculating on GPU
            kernel_cl.set_args(iobj, jobj, eta)
            kernel_cl.run()
            res['cl'] = kernel_cl.get_result()

            # calculating deviation of result
            deviation0 = np.sqrt(
                sum((c_res-cl_res)**2
                    for c_res, cl_res in zip(res['c'][0], res['cl'][0]))
            )
            deviation1 = np.sqrt(
                sum((c_res-cl_res)**2
                    for c_res, cl_res in zip(res['c'][1], res['cl'][1]))
            )
            deviations.append([deviation0.max(), deviation1.max()])

        jobj = self.small_system
        for i in range(1, npart+1):
            res = {'c': None, 'cl': None}

            # setup data
            iobj = self.small_system[:i]

            # calculating on CPU
            kernel_c.set_args(iobj, jobj, eta)
            kernel_c.run()
            res['c'] = kernel_c.get_result()

            # calculating on GPU
            kernel_cl.set_args(iobj, jobj, eta)
            kernel_cl.run()
            res['cl'] = kernel_cl.get_result()

            # calculating deviation of result
            deviation0 = np.sqrt(
                sum((c_res-cl_res)**2
                    for c_res, cl_res in zip(res['c'][0], res['cl'][0]))
            )
            deviation1 = np.sqrt(
                sum((c_res-cl_res)**2
                    for c_res, cl_res in zip(res['c'][1], res['cl'][1]))
            )
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

        kernel_c = gravity.PNAcc("c", ctype.prec)
        kernel_cl = gravity.PNAcc("cl", ctype.prec)

        iobj = self.small_system
        for j in range(1, npart+1):
            res = {'c': None, 'cl': None}

            # setup data
            jobj = self.small_system[:j]

            # calculating on CPU
            kernel_c.set_args(iobj, jobj)
            kernel_c.run()
            res['c'] = kernel_c.get_result()

            # calculating on GPU
            kernel_cl.set_args(iobj, jobj)
            kernel_cl.run()
            res['cl'] = kernel_cl.get_result()

            # calculating deviation of result
            deviation = np.sqrt(
                sum((c_res-cl_res)**2
                    for c_res, cl_res in zip(res['c'], res['cl']))
            )
            deviations.append(deviation.max())

        jobj = self.small_system
        for i in range(1, npart+1):
            res = {'c': None, 'cl': None}

            # setup data
            iobj = self.small_system[:i]

            # calculating on CPU
            kernel_c.set_args(iobj, jobj)
            kernel_c.run()
            res['c'] = kernel_c.get_result()

            # calculating on GPU
            kernel_cl.set_args(iobj, jobj)
            kernel_cl.run()
            res['cl'] = kernel_cl.get_result()

            # calculating deviation of result
            deviation = np.sqrt(
                sum((c_res-cl_res)**2
                    for c_res, cl_res in zip(res['c'], res['cl']))
            )
            deviations.append(deviation.max())

        deviations = np.array(deviations)
        print(deviations.max())

    def test06(self):
        print(
            "\ntest06: C(CPU) vs CL(device): max deviation of grav-sakura "
            "among all combinations of i- and j-particles:",
            end=" "
        )

        dt = 1.0/64
        npart = self.small_system.n
        deviations = []

        kernel_c = gravity.Sakura("c", ctype.prec)
        kernel_cl = gravity.Sakura("cl", ctype.prec)

        iobj = self.small_system
        for j in range(1, npart+1):
            res = {'c': None, 'cl': None}

            # setup data
            jobj = self.small_system[:j]

            # calculating on CPU
            kernel_c.set_args(iobj, jobj, dt)
            kernel_c.run()
            res['c'] = kernel_c.get_result()

            # calculating on GPU
            kernel_cl.set_args(iobj, jobj, dt)
            kernel_cl.run()
            res['cl'] = kernel_cl.get_result()

            # calculating deviation of result
            deviation = np.sqrt(
                sum((c_res-cl_res)**2
                    for c_res, cl_res in zip(res['c'], res['cl']))
            )
            deviations.append(deviation.max())

        jobj = self.small_system
        for i in range(1, npart+1):
            res = {'c': None, 'cl': None}

            # setup data
            iobj = self.small_system[:i]

            # calculating on CPU
            kernel_c.set_args(iobj, jobj, dt)
            kernel_c.run()
            res['c'] = kernel_c.get_result()

            # calculating on GPU
            kernel_cl.set_args(iobj, jobj, dt)
            kernel_cl.run()
            res['cl'] = kernel_cl.get_result()

            # calculating deviation of result
            deviation = np.sqrt(
                sum((c_res-cl_res)**2
                    for c_res, cl_res in zip(res['c'], res['cl']))
            )
            deviations.append(deviation.max())

        deviations = np.array(deviations)
        print(deviations.max())

    def test07(self):
        print(
            "\ntest07: C(CPU) vs CL(device): performance of grav-phi:",
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

    def test08(self):
        print(
            "\ntest08: C(CPU) vs CL(device): performance of grav-acc:",
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

    def test09(self):
        print(
            "\ntest09: C(CPU) vs CL(device): performance of grav-acc_jerk:",
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
        kernel = gravity.AccJerk("c", ctype.prec)
        best['set_args']['c'] = best_of(n, kernel.set_args, iobj, jobj)
        best['run']['c'] = best_of(n, kernel.run)
        best['get_result']['c'] = best_of(n, kernel.get_result)

        # calculating using DP on CPU
        kernel = gravity.AccJerk("cl", ctype.prec)
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

    def test10(self):
        print(
            "\ntest10: C(CPU) vs CL(device): performance of grav-tstep:",
            end=" "
        )

        n = 5   # no. of samples
        best = {'set_args': {'c': None, 'cl': None},
                'run': {'c': None, 'cl': None},
                'get_result': {'c': None, 'cl': None}}

        # setup data
        iobj = self.large_system
        jobj = self.large_system
        eta = 1.0/64

        # calculating using SP on CPU
        kernel = gravity.Tstep("c", ctype.prec)
        best['set_args']['c'] = best_of(n, kernel.set_args, iobj, jobj, eta)
        best['run']['c'] = best_of(n, kernel.run)
        best['get_result']['c'] = best_of(n, kernel.get_result)

        # calculating using DP on CPU
        kernel = gravity.Tstep("cl", ctype.prec)
        best['set_args']['cl'] = best_of(n, kernel.set_args, iobj, jobj, eta)
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

    def test11(self):
        print(
            "\ntest11: C(CPU) vs CL(device): performance of grav-pnacc:",
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

    def test12(self):
        print(
            "\ntest12: C(CPU) vs CL(device): performance of grav-sakura:",
            end=" "
        )

        n = 5   # no. of samples
        best = {'set_args': {'c': None, 'cl': None},
                'run': {'c': None, 'cl': None},
                'get_result': {'c': None, 'cl': None}}

        # setup data
        iobj = self.large_system
        jobj = self.large_system
        dt = 1.0/64

        # calculating using SP on CPU
        kernel = gravity.Sakura("c", ctype.prec)
        best['set_args']['c'] = best_of(n, kernel.set_args, iobj, jobj, dt)
        best['run']['c'] = best_of(n, kernel.run)
        best['get_result']['c'] = best_of(n, kernel.get_result)

        # calculating using DP on CPU
        kernel = gravity.Sakura("cl", ctype.prec)
        best['set_args']['cl'] = best_of(n, kernel.set_args, iobj, jobj, dt)
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
