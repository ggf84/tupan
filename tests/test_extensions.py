# -*- coding: utf-8 -*-
#

"""
Test suite for extensions module.
"""

import unittest
import numpy as np
from tupan.units import ureg
from tupan.integrator.sakura import sakura_step
from tupan.particles.system import ParticleSystem
from tupan.lib import extensions as ext

ureg.define('uM = M_sun')   # set some mass unit

setattr(ParticleSystem, 'set_sakura', sakura_step)


def set_particles(n):
    ps = ParticleSystem(n)
    b = ps.bodies
    b.mass[...] = np.random.random((n,)) * ureg('uM')
    b.pos[...] = np.random.random((3, n)) * 10 * ureg('uL')
    b.vel[...] = np.random.random((3, n)) * 10 * ureg('uL/uT')
    b.register_attribute('pnacc', '{nd}, {nb}', 'real_t', 'uL/uT**2')
    b.register_attribute('dpos', '{nd}, {nb}', 'real_t', 'uL')
    b.register_attribute('dvel', '{nd}, {nb}', 'real_t', 'uL/uT')
    return ps


def run_and_compare(ip_c, jp_c, ip_cl, jp_cl,
                    func, ckernel, clkernel, **kwargs):
    devs = []
    func(ip_c, jp_c, kernel=ckernel, **kwargs)
    func(ip_cl, jp_cl, kernel=clkernel, **kwargs)

    for c_ps, cl_ps in zip(ip_c.members.values(), ip_cl.members.values()):
        arrays_c = [getattr(c_ps, attr) for attr in c_ps.attrs]
        arrays_cl = [getattr(cl_ps, attr) for attr in cl_ps.attrs]
        for a, b in zip(arrays_c, arrays_cl):
            devs.append(abs(a - b).max().m)
    return devs


def compare_result(test_number, name, **kwargs):
    np.random.seed(0)
    ckernel = ext.make_extension(name, backend='C')
    clkernel = ext.make_extension(name, backend='CL')
    func = getattr(ParticleSystem, 'set_'+name.lower())

    ips_c, jps_c = set_particles(32), set_particles(2048)
    ips_cl, jps_cl = ips_c.copy(), jps_c.copy()

    devs = []
    devs += run_and_compare(ips_c, ips_c, ips_cl, ips_cl,
                            func, ckernel, clkernel, **kwargs)

    devs += run_and_compare(jps_c, jps_c, jps_cl, jps_cl,
                            func, ckernel, clkernel, **kwargs)

    devs += run_and_compare(ips_c, jps_c, ips_cl, jps_cl,
                            func, ckernel, clkernel, **kwargs)

    devs += run_and_compare(jps_c, ips_c, jps_cl, ips_cl,
                            func, ckernel, clkernel, **kwargs)

    print(f"\ntest{test_number:02d}: maxdev({name}): {max(devs)}")


class TestCase1(unittest.TestCase):
    """

    """
    @classmethod
    def setUpClass(cls):
        print("\n" + cls.__name__ + ": "
              "compare results calculated using C / CL extensions.")

    def test01(self):
        compare_result(1, 'Phi', nforce=1)

    def test02(self):
        compare_result(2, 'Acc', nforce=1)

    def test03(self):
        compare_result(3, 'Acc_Jrk', nforce=2)

    def test04(self):
        compare_result(4, 'Snp_Crk', nforce=4)

    def test05(self):
        eta = 1.0/64
        compare_result(5, 'Tstep', eta=eta, nforce=2)

    def test06(self):
        pn = {'order': 7, 'clight': 128.0}
        compare_result(6, 'PNAcc', pn=pn, nforce=2)

    def test07(self):
        dt = 1.0/64
        compare_result(7, 'Sakura', dt=dt, flag=-2, nforce=2)


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
