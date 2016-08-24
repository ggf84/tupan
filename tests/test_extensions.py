# -*- coding: utf-8 -*-
#

"""
Test suite for extensions module.
"""

from __future__ import print_function
import unittest
import numpy as np
from tupan.lib import extensions as ext
from tupan.lib.utils.timing import Timer


def set_particles(n):
    from tupan.particles.system import ParticleSystem

    ps = ParticleSystem(n)

    ps.eps2[...] = np.zeros((n,))
    ps.mass[...] = np.random.random((n,))
    ps.rdot[0][...] = np.random.random((3, n)) * 10
    ps.rdot[1][...] = np.random.random((3, n)) * 10
    ps.register_attribute('pnacc', '{nd}, {nb}', 'real_t')

    return ps


def compare_result(test_number, kernel_name, **kwargs):
    np.random.seed(0)
    Ckernel = ext.get_kernel(kernel_name, backend='C')
    CLkernel = ext.get_kernel(kernel_name, backend='CL')

    deviations = []
    c_ips, c_jps = set_particles(32), set_particles(2048)
    cl_ips, cl_jps = c_ips.copy(), c_jps.copy()
    for (c_ip, c_jp), (cl_ip, cl_jp) in [((c_ips, c_ips), (cl_ips, cl_ips)),
                                         ((c_ips, c_jps), (cl_ips, cl_jps)),
                                         ((c_jps, c_ips), (cl_jps, cl_ips)),
                                         ((c_jps, c_jps), (cl_jps, cl_jps))]:

        res = [Ckernel(c_ip, c_jp, **kwargs), CLkernel(cl_ip, cl_jp, **kwargs)]

        for vc, vcl in zip(*res):
            for attr in vc.attr_names:
                array_c = getattr(vc, attr)
                array_cl = getattr(vcl, attr)
                dev = abs(array_c - array_cl).max()
                deviations.append(dev)

    msg = "\ntest{0:02d}: maxdev({1}): {2}"
    print(msg.format(test_number, kernel_name, max(deviations)))


def benchmark(test_number, kernel_name, imax=12, **kwargs):
    def best_of(n, func, *args, **kwargs):
        timer = Timer()
        elapsed = []
        for i in range(n):
            timer.start()
            func(*args, **kwargs)
            elapsed.append(timer.elapsed())
        return min(elapsed)

    np.random.seed(0)
    Ckernel = getattr(ext, kernel_name+'_rectangle')(backend='C')
    CLkernel = getattr(ext, kernel_name)(backend='CL')

    msg = "\ntest{0:02d}: {1}"
    print(msg.format(test_number, kernel_name))

    for (ips, jps) in [(set_particles(2**(i+1)), set_particles(2**(i+1)))
                       for i in range(imax)]:
        smalln = ips.n <= 2**12  # 4096

        print("  N={0}:".format(ips.n))

        res = [
            best_of(5, Ckernel.set_args, ips, jps, **kwargs) if smalln else 0,
            best_of(5, CLkernel.set_args, ips, jps, **kwargs)
        ]
        ratio = res[0] / res[1]
        print("    {meth} time (s): 'C': {res[0]:.4e},"
              " 'CL': {res[1]:.4e} | ratio(C/CL): {ratio:.4f}"
              .format(meth='set', res=res, ratio=ratio))

        res = [
            best_of(3, Ckernel.run) if smalln else 0,
            best_of(3, CLkernel.run)
        ]
        ratio = res[0] / res[1]
        print("    {meth} time (s): 'C': {res[0]:.4e},"
              " 'CL': {res[1]:.4e} | ratio(C/CL): {ratio:.4f}"
              .format(meth='run', res=res, ratio=ratio))

        res = [
            best_of(5, Ckernel.map_buffers) if smalln else 0,
            best_of(5, CLkernel.map_buffers)
        ]
        ratio = res[0] / res[1]
        print("    {meth} time (s): 'C': {res[0]:.4e},"
              " 'CL': {res[1]:.4e} | ratio(C/CL): {ratio:.4f}"
              .format(meth='get', res=res, ratio=ratio))


class TestCase1(unittest.TestCase):
    """

    """
    @classmethod
    def setUpClass(cls):
        print("\n" + cls.__name__ + ": "
              "compare results calculated using C / CL extensions.")

    def test01(self):
        compare_result(1, 'Phi')

    def test02(self):
        compare_result(2, 'Acc')

    def test03(self):
        compare_result(3, 'AccJrk')

    def test04(self):
        compare_result(4, 'SnpCrk')

    def test05(self):
        eta = 1.0/64
        compare_result(5, 'Tstep', eta=eta)

    def test06(self):
        pn = {'order': 7, 'clight': 128.0}
        compare_result(6, 'PNAcc', pn=pn)

    def test07(self):
        dt = 1.0/64
        compare_result(7, 'Sakura', dt=dt, flag=-2)

    def test08(self):
        dt = 1.0/64
        compare_result(8, 'NregX', dt=dt)

    def test09(self):
        dt = 1.0/64
        compare_result(9, 'NregV', dt=dt)


class TestCase2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n" + cls.__name__ + ": "
              "benchmark kernels using C / CL extensions.")

    def test01(self):
        benchmark(1, 'Phi')

    def test02(self):
        benchmark(2, 'Acc')

    def test03(self):
        benchmark(3, 'AccJrk')

    def test04(self):
        benchmark(4, 'SnpCrk')

    def test05(self):
        eta = 1.0/64
        benchmark(5, 'Tstep', eta=eta)

    def test06(self):
        pn = {'order': 7, 'clight': 128.0}
        benchmark(6, 'PNAcc', pn=pn)

    def test07(self):
        dt = 1.0/64
        benchmark(7, 'Sakura', dt=dt, flag=-2)

    def test08(self):
        dt = 1.0/64
        benchmark(8, 'NregX', dt=dt)

    def test09(self):
        dt = 1.0/64
        benchmark(9, 'NregV', dt=dt)


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


# -- End of File --
