# -*- coding: utf-8 -*-
#

"""
Test suite for extensions module.
"""

import unittest
import numpy as np
from tupan.lib import extensions as ext


def set_particles(n):
    from tupan.particles.system import ParticleSystem

    ps = ParticleSystem(n)
    b = ps.bodies
    b.mass[...] = np.random.random((n,))
    b.rdot[0][...] = np.random.random((3, n)) * 10
    b.rdot[1][...] = np.random.random((3, n)) * 10
    b.register_attribute('pnacc', '{nd}, {nb}', 'real_t')
    b.register_attribute('drdot', '2, {nd}, {nb}', 'real_t')
    return b


def get_kernel(name, backend):
    kernel_r = ext.make_extension(name+'_rectangle', backend)
    kernel_t = ext.make_extension(name+'_triangle', backend)

    def func(ips, jps, *args, **kwargs):
        if ips != jps:
            return kernel_r(ips, jps, *args, **kwargs)
        return kernel_t(ips, *args, **kwargs)

    return func


def compare_result(test_number, kernel_name, **kwargs):
    np.random.seed(0)
    Ckernel = get_kernel(kernel_name, backend='C')
    CLkernel = get_kernel(kernel_name, backend='CL')

    deviations = []
    c_ips, c_jps = set_particles(32), set_particles(2048)
    cl_ips, cl_jps = c_ips.copy(), c_jps.copy()
    for (c_ip, c_jp), (cl_ip, cl_jp) in [((c_ips, c_ips), (cl_ips, cl_ips)),
                                         ((c_jps, c_jps), (cl_jps, cl_jps)),
                                         ((c_ips, c_jps), (cl_ips, cl_jps)),
                                         ((c_jps, c_ips), (cl_jps, cl_ips)), ]:

        res = [Ckernel(c_ip, c_jp, **kwargs), CLkernel(cl_ip, cl_jp, **kwargs)]

        for vc, vcl in zip(*res):
            for attr in vc.attrs:
                array_c = getattr(vc, attr)
                array_cl = getattr(vcl, attr)
                dev = abs(array_c - array_cl).max()
                deviations.append(dev)

    msg = "\ntest{0:02d}: maxdev({1}): {2}"
    print(msg.format(test_number, kernel_name, max(deviations)))


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
        compare_result(2, 'Acc', nforce=1)

    def test03(self):
        compare_result(3, 'AccJrk', nforce=2)

    def test04(self):
        compare_result(4, 'SnpCrk', nforce=4)

    def test05(self):
        eta = 1.0/64
        compare_result(5, 'Tstep', eta=eta)

    def test06(self):
        pn = {'order': 7, 'clight': 128.0}
        compare_result(6, 'PNAcc', pn=pn)

    def test07(self):
        dt = 1.0/64
        compare_result(7, 'Sakura', dt=dt, flag=-2)


if __name__ == "__main__":
    def load_tests(test_cases):
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        for test_class in test_cases:
            tests = loader.loadTestsFromTestCase(test_class)
            suite.addTests(tests)
        return suite

    test_cases = (TestCase1,)

    suite = load_tests(test_cases)
    unittest.TextTestRunner(verbosity=1, failfast=True).run(suite)


# -- End of File --
