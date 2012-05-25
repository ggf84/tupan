#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for extensions module.
"""


from __future__ import print_function
import unittest
import numpy as np
from pynbody.lib.extensions import libkernels
from pynbody.lib.utils.timing import Timer



def set_particles(npart):
    if npart < 2: npart = 2
    from pynbody.ics.imf import IMF
    from pynbody.ics.plummer import Plummer
    imf = IMF.padoan2007(0.075, 120.0)
    p = Plummer(npart, imf, eps=0.0, seed=1)
    p.make_plummer()
    bi = p.particles['body']
    return bi

small_system = set_particles(64)
large_system = set_particles(4096)



class TestCase(unittest.TestCase):

    def test01(self):
        print('\ntest01: max deviation of grav-phi (in SP on CPU and GPU) between all combinations of i- and j-particles:', end=' ')

        npart = len(small_system)
        deviations = []

        for i in range(1, npart+1):
            for j in range(1, npart+1):
                phi = {'cpu_result': None, 'gpu_result': None}

                # setup data
                iobj = small_system[:i].copy()
                jobj = small_system[:j].copy()
                ni = len(iobj)
                nj = len(jobj)
                iposmass = np.concatenate((iobj.pos, iobj.mass[..., np.newaxis]), axis=1)
                jposmass = np.concatenate((jobj.pos, jobj.mass[..., np.newaxis]), axis=1)
                data = (ni, iposmass, iobj.eps2,
                        nj, jposmass, jobj.eps2)

                result_shape = (ni,)
                local_memory_shape = (4, 1)
                local_size = 384
                global_size = ((ni-1)//local_size + 1) * local_size


                # calculating on CPU
                phi_kernel = libkernels['sp']['c'].p2p_phi_kernel
                phi_kernel.set_args(*data, global_size=global_size,
                                           local_size=local_size,
                                           result_shape=result_shape,
                                           local_memory_shape=local_memory_shape)
                phi_kernel.run()
                phi['cpu_result'] = phi_kernel.get_result()


                # calculating on GPU
                phi_kernel = libkernels['sp']['cl'].p2p_phi_kernel
                phi_kernel.set_args(*data, global_size=global_size,
                                           local_size=local_size,
                                           result_shape=result_shape,
                                           local_memory_shape=local_memory_shape)
                phi_kernel.run()
                phi['gpu_result'] = phi_kernel.get_result()

                # calculating diff of result
                phi_deviation = np.abs(phi['cpu_result'] - phi['gpu_result'])
                deviations.append(phi_deviation.max())

        deviations = np.array(deviations)
        print(deviations.max())


    def test02(self):
        print('\ntest02: max deviation of grav-acc (in SP on CPU and GPU) between all combinations of i- and j-particles:', end=' ')

        npart = len(small_system)
        deviations = []

        for i in range(1, npart+1):
            for j in range(1, npart+1):
                acc = {'cpu_result': None, 'gpu_result': None}

                # setup data
                iobj = small_system[:i].copy()
                jobj = small_system[:j].copy()
                ni = len(iobj)
                nj = len(jobj)
                iposmass = np.concatenate((iobj.pos, iobj.mass[..., np.newaxis]), axis=1)
                jposmass = np.concatenate((jobj.pos, jobj.mass[..., np.newaxis]), axis=1)
                data = (ni, iposmass, iobj.eps2,
                        nj, jposmass, jobj.eps2)

                result_shape = (ni, 4)
                local_memory_shape = (4, 1)
                local_size = 384
                global_size = ((ni-1)//local_size + 1) * local_size


                # calculating on CPU
                acc_kernel = libkernels['sp']['c'].p2p_acc_kernel
                acc_kernel.set_args(*data, global_size=global_size,
                                           local_size=local_size,
                                           result_shape=result_shape,
                                           local_memory_shape=local_memory_shape)
                acc_kernel.run()
                acc['cpu_result'] = acc_kernel.get_result()[:,:3]


                # calculating on GPU
                acc_kernel = libkernels['sp']['cl'].p2p_acc_kernel
                acc_kernel.set_args(*data, global_size=global_size,
                                           local_size=local_size,
                                           result_shape=result_shape,
                                           local_memory_shape=local_memory_shape)
                acc_kernel.run()
                acc['gpu_result'] = acc_kernel.get_result()[:,:3]

                # calculating diff of result
                acc_deviation = np.sqrt(((acc['cpu_result']-acc['gpu_result'])**2).sum(1))
                deviations.append(acc_deviation.max())

        deviations = np.array(deviations)
        print(deviations.max())


    def test03(self):
        print('\ntest03: max deviation of grav-pnacc (in SP on CPU and GPU) between all combinations of i- and j-particles:', end=' ')

        npart = len(small_system)
        deviations = []

        for i in range(1, npart+1):
            for j in range(1, npart+1):
                pnacc = {'cpu_result': None, 'gpu_result': None}

                # setup data
                iobj = small_system[:i].copy()
                jobj = small_system[:j].copy()
                ni = len(iobj)
                nj = len(jobj)
                iposmass = np.concatenate((iobj.pos, iobj.mass[..., np.newaxis]), axis=1)
                jposmass = np.concatenate((jobj.pos, jobj.mass[..., np.newaxis]), axis=1)
                iveliv2 = np.concatenate((iobj.vel, (iobj.vel**2).sum(1)[..., np.newaxis]), axis=1)
                jveljv2 = np.concatenate((jobj.vel, (jobj.vel**2).sum(1)[..., np.newaxis]), axis=1)
                from pynbody.lib.gravity import Clight
                clight = Clight(7, 128)
                data = (ni, iposmass, iveliv2,
                        nj, jposmass, jveljv2,
                        clight.pn_order, clight.inv1,
                        clight.inv2, clight.inv3,
                        clight.inv4, clight.inv5,
                        clight.inv6, clight.inv7)

                result_shape = (ni, 4)
                local_memory_shape = (4, 4)
                local_size = 384
                global_size = ((ni-1)//local_size + 1) * local_size


                # calculating on CPU
                pnacc_kernel = libkernels['sp']['c'].p2p_pnacc_kernel
                pnacc_kernel.set_args(*data, global_size=global_size,
                                             local_size=local_size,
                                             result_shape=result_shape,
                                             local_memory_shape=local_memory_shape)
                pnacc_kernel.run()
                pnacc['cpu_result'] = pnacc_kernel.get_result()[:,:3]


                # calculating on GPU
                pnacc_kernel = libkernels['sp']['cl'].p2p_pnacc_kernel
                pnacc_kernel.set_args(*data, global_size=global_size,
                                             local_size=local_size,
                                             result_shape=result_shape,
                                             local_memory_shape=local_memory_shape)
                pnacc_kernel.run()
                pnacc['gpu_result'] = pnacc_kernel.get_result()[:,:3]

                # calculating diff of result
                pnacc_deviation = np.sqrt(((pnacc['cpu_result']-pnacc['gpu_result'])**2).sum(1))
                deviations.append(pnacc_deviation.max())

        deviations = np.array(deviations)
        print(deviations.max())


    def test04(self):
        print('\ntest04: max deviation of grav-phi (in DP on CPU and GPU) between all combinations of i- and j-particles:', end=' ')

        npart = len(small_system)
        deviations = []

        for i in range(1, npart+1):
            for j in range(1, npart+1):
                phi = {'cpu_result': None, 'gpu_result': None}

                # setup data
                iobj = small_system[:i].copy()
                jobj = small_system[:j].copy()
                ni = len(iobj)
                nj = len(jobj)
                iposmass = np.concatenate((iobj.pos, iobj.mass[..., np.newaxis]), axis=1)
                jposmass = np.concatenate((jobj.pos, jobj.mass[..., np.newaxis]), axis=1)
                data = (ni, iposmass, iobj.eps2,
                        nj, jposmass, jobj.eps2)

                result_shape = (ni,)
                local_memory_shape = (4, 1)
                local_size = 384
                global_size = ((ni-1)//local_size + 1) * local_size


                # calculating on CPU
                phi_kernel = libkernels['dp']['c'].p2p_phi_kernel
                phi_kernel.set_args(*data, global_size=global_size,
                                           local_size=local_size,
                                           result_shape=result_shape,
                                           local_memory_shape=local_memory_shape)
                phi_kernel.run()
                phi['cpu_result'] = phi_kernel.get_result()


                # calculating on GPU
                phi_kernel = libkernels['dp']['cl'].p2p_phi_kernel
                phi_kernel.set_args(*data, global_size=global_size,
                                           local_size=local_size,
                                           result_shape=result_shape,
                                           local_memory_shape=local_memory_shape)
                phi_kernel.run()
                phi['gpu_result'] = phi_kernel.get_result()

                # calculating diff of result
                phi_deviation = np.abs(phi['cpu_result'] - phi['gpu_result'])
                deviations.append(phi_deviation.max())

        deviations = np.array(deviations)
        print(deviations.max())


    def test05(self):
        print('\ntest05: max deviation of grav-acc (in DP on CPU and GPU) between all combinations of i- and j-particles:', end=' ')

        npart = len(small_system)
        deviations = []

        for i in range(1, npart+1):
            for j in range(1, npart+1):
                acc = {'cpu_result': None, 'gpu_result': None}

                # setup data
                iobj = small_system[:i].copy()
                jobj = small_system[:j].copy()
                ni = len(iobj)
                nj = len(jobj)
                iposmass = np.concatenate((iobj.pos, iobj.mass[..., np.newaxis]), axis=1)
                jposmass = np.concatenate((jobj.pos, jobj.mass[..., np.newaxis]), axis=1)
                data = (ni, iposmass, iobj.eps2,
                        nj, jposmass, jobj.eps2)

                result_shape = (ni, 4)
                local_memory_shape = (4, 1)
                local_size = 384
                global_size = ((ni-1)//local_size + 1) * local_size


                # calculating on CPU
                acc_kernel = libkernels['dp']['c'].p2p_acc_kernel
                acc_kernel.set_args(*data, global_size=global_size,
                                           local_size=local_size,
                                           result_shape=result_shape,
                                           local_memory_shape=local_memory_shape)
                acc_kernel.run()
                acc['cpu_result'] = acc_kernel.get_result()[:,:3]


                # calculating on GPU
                acc_kernel = libkernels['dp']['cl'].p2p_acc_kernel
                acc_kernel.set_args(*data, global_size=global_size,
                                           local_size=local_size,
                                           result_shape=result_shape,
                                           local_memory_shape=local_memory_shape)
                acc_kernel.run()
                acc['gpu_result'] = acc_kernel.get_result()[:,:3]

                # calculating diff of result
                acc_deviation = np.sqrt(((acc['cpu_result']-acc['gpu_result'])**2).sum(1))
                deviations.append(acc_deviation.max())

        deviations = np.array(deviations)
        print(deviations.max())


    def test06(self):
        print('\ntest06: max deviation of grav-pnacc (in DP on CPU and GPU) between all combinations of i- and j-particles:', end=' ')

        npart = len(small_system)
        deviations = []

        for i in range(1, npart+1):
            for j in range(1, npart+1):
                pnacc = {'cpu_result': None, 'gpu_result': None}

                # setup data
                iobj = small_system[:i].copy()
                jobj = small_system[:j].copy()
                ni = len(iobj)
                nj = len(jobj)
                iposmass = np.concatenate((iobj.pos, iobj.mass[..., np.newaxis]), axis=1)
                jposmass = np.concatenate((jobj.pos, jobj.mass[..., np.newaxis]), axis=1)
                iveliv2 = np.concatenate((iobj.vel, (iobj.vel**2).sum(1)[..., np.newaxis]), axis=1)
                jveljv2 = np.concatenate((jobj.vel, (jobj.vel**2).sum(1)[..., np.newaxis]), axis=1)
                from pynbody.lib.gravity import Clight
                clight = Clight(7, 128)
                data = (ni, iposmass, iveliv2,
                        nj, jposmass, jveljv2,
                        clight.pn_order, clight.inv1,
                        clight.inv2, clight.inv3,
                        clight.inv4, clight.inv5,
                        clight.inv6, clight.inv7)

                result_shape = (ni, 4)
                local_memory_shape = (4, 4)
                local_size = 384
                global_size = ((ni-1)//local_size + 1) * local_size


                # calculating on CPU
                pnacc_kernel = libkernels['dp']['c'].p2p_pnacc_kernel
                pnacc_kernel.set_args(*data, global_size=global_size,
                                             local_size=local_size,
                                             result_shape=result_shape,
                                             local_memory_shape=local_memory_shape)
                pnacc_kernel.run()
                pnacc['cpu_result'] = pnacc_kernel.get_result()[:,:3]


                # calculating on GPU
                pnacc_kernel = libkernels['dp']['cl'].p2p_pnacc_kernel
                pnacc_kernel.set_args(*data, global_size=global_size,
                                             local_size=local_size,
                                             result_shape=result_shape,
                                             local_memory_shape=local_memory_shape)
                pnacc_kernel.run()
                pnacc['gpu_result'] = pnacc_kernel.get_result()[:,:3]

                # calculating diff of result
                pnacc_deviation = np.sqrt(((pnacc['cpu_result']-pnacc['gpu_result'])**2).sum(1))
                deviations.append(pnacc_deviation.max())

        deviations = np.array(deviations)
        print(deviations.max())


    def test07(self):
        print('\ntest07: performance of grav-phi (in SP and DP on CPU):', end=' ')

        nsamples = 5
        timer = Timer()

        timings = {'cpu_single': None, 'cpu_double': None}

        # setup data
        iobj = large_system.copy()
        jobj = large_system.copy()
        ni = len(iobj)
        nj = len(jobj)
        iposmass = np.concatenate((iobj.pos, iobj.mass[..., np.newaxis]), axis=1)
        jposmass = np.concatenate((jobj.pos, jobj.mass[..., np.newaxis]), axis=1)
        data = (ni, iposmass, iobj.eps2,
                nj, jposmass, jobj.eps2)

        result_shape = (ni,)
        local_memory_shape = (4, 1)
        local_size = 384
        global_size = ((ni-1)//local_size + 1) * local_size


        # calculating using SP on CPU
        phi_kernel = libkernels['sp']['c'].p2p_phi_kernel
        phi_kernel.set_args(*data, global_size=global_size,
                                   local_size=local_size,
                                   result_shape=result_shape,
                                   local_memory_shape=local_memory_shape)
        elapsed_sum = 0.0
        for i in range(nsamples):
            timer.start()
            phi_kernel.run()
            elapsed_sum += timer.elapsed()
        ret = phi_kernel.get_result()
        timings['cpu_single'] = elapsed_sum / nsamples


        # calculating using DP on CPU
        phi_kernel = libkernels['dp']['c'].p2p_phi_kernel
        phi_kernel.set_args(*data, global_size=global_size,
                                   local_size=local_size,
                                   result_shape=result_shape,
                                   local_memory_shape=local_memory_shape)
        elapsed_sum = 0.0
        for i in range(nsamples):
            timer.start()
            phi_kernel.run()
            elapsed_sum += timer.elapsed()
        ret = phi_kernel.get_result()
        timings['cpu_double'] = elapsed_sum / nsamples

        print(timings)


    def test08(self):
        print('\ntest08: performance of grav-acc (in SP and DP on CPU):', end=' ')

        nsamples = 5
        timer = Timer()

        timings = {'cpu_single': None, 'cpu_double': None}

        # setup data
        iobj = large_system.copy()
        jobj = large_system.copy()
        ni = len(iobj)
        nj = len(jobj)
        iposmass = np.concatenate((iobj.pos, iobj.mass[..., np.newaxis]), axis=1)
        jposmass = np.concatenate((jobj.pos, jobj.mass[..., np.newaxis]), axis=1)
        data = (ni, iposmass, iobj.eps2,
                nj, jposmass, jobj.eps2)

        result_shape = (ni, 4)
        local_memory_shape = (4, 1)
        local_size = 384
        global_size = ((ni-1)//local_size + 1) * local_size


        # calculating using SP on CPU
        acc_kernel = libkernels['sp']['c'].p2p_acc_kernel
        acc_kernel.set_args(*data, global_size=global_size,
                                   local_size=local_size,
                                   result_shape=result_shape,
                                   local_memory_shape=local_memory_shape)
        elapsed_sum = 0.0
        for i in range(nsamples):
            timer.start()
            acc_kernel.run()
            elapsed_sum += timer.elapsed()
        ret = acc_kernel.get_result()
        timings['cpu_single'] = elapsed_sum / nsamples


        # calculating using DP on CPU
        acc_kernel = libkernels['dp']['c'].p2p_acc_kernel
        acc_kernel.set_args(*data, global_size=global_size,
                                   local_size=local_size,
                                   result_shape=result_shape,
                                   local_memory_shape=local_memory_shape)
        elapsed_sum = 0.0
        for i in range(nsamples):
            timer.start()
            acc_kernel.run()
            elapsed_sum += timer.elapsed()
        ret = acc_kernel.get_result()
        timings['cpu_double'] = elapsed_sum / nsamples

        print(timings)


    def test09(self):
        print('\ntest09: performance of grav-pnacc (in SP and DP on CPU):', end=' ')

        nsamples = 5
        timer = Timer()

        timings = {'cpu_single': None, 'cpu_double': None}

        # setup data
        iobj = large_system.copy()
        jobj = large_system.copy()
        ni = len(iobj)
        nj = len(jobj)
        iposmass = np.concatenate((iobj.pos, iobj.mass[..., np.newaxis]), axis=1)
        jposmass = np.concatenate((jobj.pos, jobj.mass[..., np.newaxis]), axis=1)
        iveliv2 = np.concatenate((iobj.vel, (iobj.vel**2).sum(1)[..., np.newaxis]), axis=1)
        jveljv2 = np.concatenate((jobj.vel, (jobj.vel**2).sum(1)[..., np.newaxis]), axis=1)
        from pynbody.lib.gravity import Clight
        clight = Clight(7, 128)
        data = (ni, iposmass, iveliv2,
                nj, jposmass, jveljv2,
                clight.pn_order, clight.inv1,
                clight.inv2, clight.inv3,
                clight.inv4, clight.inv5,
                clight.inv6, clight.inv7)

        result_shape = (ni, 4)
        local_memory_shape = (4, 4)
        local_size = 384
        global_size = ((ni-1)//local_size + 1) * local_size


        # calculating using SP on CPU
        pnacc_kernel = libkernels['sp']['c'].p2p_pnacc_kernel
        pnacc_kernel.set_args(*data, global_size=global_size,
                                     local_size=local_size,
                                     result_shape=result_shape,
                                     local_memory_shape=local_memory_shape)
        elapsed_sum = 0.0
        for i in range(nsamples):
            timer.start()
            pnacc_kernel.run()
            elapsed_sum += timer.elapsed()
        ret = pnacc_kernel.get_result()
        timings['cpu_single'] = elapsed_sum / nsamples


        # calculating using DP on CPU
        pnacc_kernel = libkernels['dp']['c'].p2p_pnacc_kernel
        pnacc_kernel.set_args(*data, global_size=global_size,
                                     local_size=local_size,
                                     result_shape=result_shape,
                                     local_memory_shape=local_memory_shape)
        elapsed_sum = 0.0
        for i in range(nsamples):
            timer.start()
            pnacc_kernel.run()
            elapsed_sum += timer.elapsed()
        ret = pnacc_kernel.get_result()
        timings['cpu_double'] = elapsed_sum / nsamples

        print(timings)


    def test10(self):
        print('\ntest10: performance of grav-phi (in SP and DP on GPU):', end=' ')

        nsamples = 5
        timer = Timer()

        timings = {'gpu_single': None, 'gpu_double': None}

        # setup data
        iobj = large_system.copy()
        jobj = large_system.copy()
        ni = len(iobj)
        nj = len(jobj)
        iposmass = np.concatenate((iobj.pos, iobj.mass[..., np.newaxis]), axis=1)
        jposmass = np.concatenate((jobj.pos, jobj.mass[..., np.newaxis]), axis=1)
        data = (ni, iposmass, iobj.eps2,
                nj, jposmass, jobj.eps2)

        result_shape = (ni,)
        local_memory_shape = (4, 1)
        local_size = 384
        global_size = ((ni-1)//local_size + 1) * local_size


        # calculating using SP on GPU
        phi_kernel = libkernels['sp']['cl'].p2p_phi_kernel
        phi_kernel.set_args(*data, global_size=global_size,
                                   local_size=local_size,
                                   result_shape=result_shape,
                                   local_memory_shape=local_memory_shape)
        elapsed_sum = 0.0
        for i in range(nsamples):
            timer.start()
            phi_kernel.run()
            elapsed_sum += timer.elapsed()
        ret = phi_kernel.get_result()
        timings['gpu_single'] = elapsed_sum / nsamples


        # calculating using DP on GPU
        phi_kernel = libkernels['dp']['cl'].p2p_phi_kernel
        phi_kernel.set_args(*data, global_size=global_size,
                                   local_size=local_size,
                                   result_shape=result_shape,
                                   local_memory_shape=local_memory_shape)
        elapsed_sum = 0.0
        for i in range(nsamples):
            timer.start()
            phi_kernel.run()
            elapsed_sum += timer.elapsed()
        ret = phi_kernel.get_result()
        timings['gpu_double'] = elapsed_sum / nsamples

        print(timings)


    def test11(self):
        print('\ntest11: performance of grav-acc (in SP and DP on GPU):', end=' ')

        nsamples = 5
        timer = Timer()

        timings = {'gpu_single': None, 'gpu_double': None}

        # setup data
        iobj = large_system.copy()
        jobj = large_system.copy()
        ni = len(iobj)
        nj = len(jobj)
        iposmass = np.concatenate((iobj.pos, iobj.mass[..., np.newaxis]), axis=1)
        jposmass = np.concatenate((jobj.pos, jobj.mass[..., np.newaxis]), axis=1)
        data = (ni, iposmass, iobj.eps2,
                nj, jposmass, jobj.eps2)

        result_shape = (ni, 4)
        local_memory_shape = (4, 1)
        local_size = 384
        global_size = ((ni-1)//local_size + 1) * local_size


        # calculating using SP on GPU
        acc_kernel = libkernels['sp']['cl'].p2p_acc_kernel
        acc_kernel.set_args(*data, global_size=global_size,
                                   local_size=local_size,
                                   result_shape=result_shape,
                                   local_memory_shape=local_memory_shape)
        elapsed_sum = 0.0
        for i in range(nsamples):
            timer.start()
            acc_kernel.run()
            elapsed_sum += timer.elapsed()
        ret = acc_kernel.get_result()
        timings['gpu_single'] = elapsed_sum / nsamples


        # calculating using DP on GPU
        acc_kernel = libkernels['dp']['cl'].p2p_acc_kernel
        acc_kernel.set_args(*data, global_size=global_size,
                                   local_size=local_size,
                                   result_shape=result_shape,
                                   local_memory_shape=local_memory_shape)
        elapsed_sum = 0.0
        for i in range(nsamples):
            timer.start()
            acc_kernel.run()
            elapsed_sum += timer.elapsed()
        ret = acc_kernel.get_result()
        timings['gpu_double'] = elapsed_sum / nsamples

        print(timings)


    def test12(self):
        print('\ntest12: performance of grav-pnacc (in SP and DP on GPU):', end=' ')

        nsamples = 5
        timer = Timer()

        timings = {'gpu_single': None, 'gpu_double': None}

        # setup data
        iobj = large_system.copy()
        jobj = large_system.copy()
        ni = len(iobj)
        nj = len(jobj)
        iposmass = np.concatenate((iobj.pos, iobj.mass[..., np.newaxis]), axis=1)
        jposmass = np.concatenate((jobj.pos, jobj.mass[..., np.newaxis]), axis=1)
        iveliv2 = np.concatenate((iobj.vel, (iobj.vel**2).sum(1)[..., np.newaxis]), axis=1)
        jveljv2 = np.concatenate((jobj.vel, (jobj.vel**2).sum(1)[..., np.newaxis]), axis=1)
        from pynbody.lib.gravity import Clight
        clight = Clight(7, 128)
        data = (ni, iposmass, iveliv2,
                nj, jposmass, jveljv2,
                clight.pn_order, clight.inv1,
                clight.inv2, clight.inv3,
                clight.inv4, clight.inv5,
                clight.inv6, clight.inv7)

        result_shape = (ni, 4)
        local_memory_shape = (4, 4)
        local_size = 384
        global_size = ((ni-1)//local_size + 1) * local_size


        # calculating using SP on GPU
        pnacc_kernel = libkernels['sp']['cl'].p2p_pnacc_kernel
        pnacc_kernel.set_args(*data, global_size=global_size,
                                     local_size=local_size,
                                     result_shape=result_shape,
                                     local_memory_shape=local_memory_shape)
        elapsed_sum = 0.0
        for i in range(nsamples):
            timer.start()
            pnacc_kernel.run()
            elapsed_sum += timer.elapsed()
        ret = pnacc_kernel.get_result()
        timings['gpu_single'] = elapsed_sum / nsamples


        # calculating using DP on GPU
        pnacc_kernel = libkernels['dp']['cl'].p2p_pnacc_kernel
        pnacc_kernel.set_args(*data, global_size=global_size,
                                     local_size=local_size,
                                     result_shape=result_shape,
                                     local_memory_shape=local_memory_shape)
        elapsed_sum = 0.0
        for i in range(nsamples):
            timer.start()
            pnacc_kernel.run()
            elapsed_sum += timer.elapsed()
        ret = pnacc_kernel.get_result()
        timings['gpu_double'] = elapsed_sum / nsamples

        print(timings)



if __name__ == "__main__":
    unittest.main()


########## end of file ##########
