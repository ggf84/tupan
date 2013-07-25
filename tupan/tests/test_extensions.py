# -*- coding: utf-8 -*-
#

"""
Test suite for extensions module.
"""


from __future__ import print_function
import sys
import unittest
from collections import OrderedDict, defaultdict
from tupan.lib import gravity
from tupan.lib.utils import ctype
from tupan.lib.utils.timing import Timer


def best_of(n, func, *args, **kwargs):
    timer = Timer()
    elapsed = []
    for i in range(n):
        timer.start()
        func(*args, **kwargs)
        elapsed.append(timer.elapsed())
    return min(elapsed)


def set_particles(n):
    import numpy as np
    from tupan.particles.allparticles import ParticleSystem
    ps = ParticleSystem(n-n//2, n//2)
    ps.mass[:] = np.random.random((ps.n,))
    ps.eps2[:] = np.zeros((ps.n,))
    ps.rx[:], ps.ry[:], ps.rz[:] = np.random.random((ps.n, 3)).T * 10
    ps.vx[:], ps.vy[:], ps.vz[:] = np.random.random((ps.n, 3)).T * 10
    ps.register_attr("ax", ctype.REAL)
    ps.register_attr("ay", ctype.REAL)
    ps.register_attr("az", ctype.REAL)
    ps.ax[:], ps.ay[:], ps.az[:] = np.random.random((ps.n, 3)).T * 100
    return ps


class TestCase1(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ps = set_particles(128)

    def compare_result(self, kernel, ps, *args):
        msg = ("gravity.{0}: max deviation of results "
               "calculated using C(CPU) vs CL(device):")
        print(msg.format(kernel.__name__))

        n = ps.n
        deviations = []

        krnlC = kernel("C", ctype.prec)
        krnlCL = kernel("CL", ctype.prec)

        iobj = ps
        for jdx in range(1, n+1):
            res = {}

            # setup data
            jobj = ps[:jdx]

            # calculating using C on CPU
            res["C"] = krnlC.calc(iobj, jobj, *args)

            # calculating using CL on device
            res["CL"] = krnlCL.calc(iobj, jobj, *args)

            # estimate deviation
            deviation = max(abs(c_res-cl_res).max()
                            for (c_res, cl_res) in zip(res["C"], res["CL"]))
            deviations.append(deviation)

            # calculating using C on CPU
            res["C"] = krnlC.calc(jobj, iobj, *args)

            # calculating using CL on device
            res["CL"] = krnlCL.calc(jobj, iobj, *args)

            # estimate deviation
            deviation = max(abs(resC - resCL).max()
                            for (resC, resCL) in zip(res["C"], res["CL"]))
            deviations.append(deviation)

        print(max(deviations))

    def test01(self):
        print("\n---------- test01 ----------")
        self.compare_result(gravity.Phi, self.ps)

    def test02(self):
        print("\n---------- test02 ----------")
        self.compare_result(gravity.Acc, self.ps)

    def test03(self):
        print("\n---------- test03 ----------")
        self.compare_result(gravity.AccJerk, self.ps)

    def test04(self):
        print("\n---------- test04 ----------")
        eta = 1.0/64
        self.compare_result(gravity.Tstep, self.ps, eta)

    def test05(self):
        print("\n---------- test05 ----------")
        gravity.clight.pn_order = 7
        gravity.clight.clight = 128
        self.compare_result(gravity.PNAcc, self.ps)

    def test06(self):
        print("\n---------- test06 ----------")
        dt = 1.0/64
        self.compare_result(gravity.Sakura, self.ps, dt)

    def test07(self):
        print("\n---------- test07 ----------")
        dt = 1.0/64
        self.compare_result(gravity.NREG_X, self.ps, dt)

    def test08(self):
        print("\n---------- test08 ----------")
        dt = 1.0/64
        self.compare_result(gravity.NREG_V, self.ps, dt)


highN = True if "--highN" in sys.argv else False


class TestCase2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        imax = 11
        if highN:
            imax = 13
        cls.pslist = [set_particles(2**(i+1)) for i in range(imax)]

    def performance(self, kernel, pslist, *args):
        msg = ("gravity.{0}: performance measurement:")
        print(msg.format(kernel.__name__))

        krnlC = kernel("C", ctype.prec)
        krnlCL = kernel("CL", ctype.prec)

        for ps in pslist:
            best = OrderedDict()
            best["set"] = defaultdict(float)
            best["get"] = defaultdict(float)
            best["run"] = defaultdict(float)

            # calculating using CL on device
            best['set']["CL"] = best_of(5, krnlCL.set_args, ps, ps, *args)
            best['run']["CL"] = best_of(3, krnlCL.run)
            best['get']["CL"] = best_of(5, krnlCL.get_result)

            if ps.n > 2048:
                print("  N={0}:".format(ps.n))
                for (k, v) in best.items():
                    print("    {k} time (s): 'C': ----------, 'CL': {CL:.4e} "
                          "| ratio(C/CL): ------".format(k=k, CL=v["CL"]))

                overhead = {}
                for k, v in best["run"].items():
                    overhead[k] = (best["set"][k] + best["get"][k]) / v
                    overhead[k] *= 100
                print("    overhead (%): 'C': ----------, "
                      "'CL': {CL:.4e}".format(CL=overhead["CL"]))
            else:
                # calculating using C on CPU
                best['set']["C"] = best_of(5, krnlC.set_args, ps, ps, *args)
                best['run']["C"] = best_of(3, krnlC.run)
                best['get']["C"] = best_of(5, krnlC.get_result)

                print("  N={0}:".format(ps.n))
                for (k, v) in best.items():
                    r = v["C"] / v["CL"]
                    print("    {k} time (s): 'C': {C:.4e}, 'CL': {CL:.4e} "
                          "| ratio(C/CL): {r:.4f}".format(k=k, r=r, **v))

                overhead = {}
                for k, v in best["run"].items():
                    overhead[k] = (best["set"][k] + best["get"][k]) / v
                    overhead[k] *= 100
                print("    overhead (%): 'C': {C:.4e}, "
                      "'CL': {CL:.4e}".format(**overhead))

    def test01(self):
        print("\n---------- test01 ----------")
        self.performance(gravity.Phi, self.pslist)

    def test02(self):
        print("\n---------- test02 ----------")
        self.performance(gravity.Acc, self.pslist)

    def test03(self):
        print("\n---------- test03 ----------")
        self.performance(gravity.AccJerk, self.pslist)

    def test04(self):
        print("\n---------- test04 ----------")
        eta = 1.0/64
        self.performance(gravity.Tstep, self.pslist, eta)

    def test05(self):
        print("\n---------- test05 ----------")
        gravity.clight.pn_order = 7
        gravity.clight.clight = 128
        self.performance(gravity.PNAcc, self.pslist)

    def test06(self):
        print("\n---------- test06 ----------")
        dt = 1.0/64
        self.performance(gravity.Sakura, self.pslist, dt)

    def test07(self):
        print("\n---------- test07 ----------")
        dt = 1.0/64
        self.performance(gravity.NREG_X, self.pslist, dt)

    def test08(self):
        print("\n---------- test08 ----------")
        dt = 1.0/64
        self.performance(gravity.NREG_V, self.pslist, dt)


if __name__ == "__main__":
    def load_tests(test_cases):
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        for test_class in test_cases:
            tests = loader.loadTestsFromTestCase(test_class)
            suite.addTests(tests)
        return suite

    test_cases = (TestCase1, TestCase2)

    suite = load_tests(test_cases)
    unittest.TextTestRunner(verbosity=1, failfast=True).run(suite)


########## end of file ##########
