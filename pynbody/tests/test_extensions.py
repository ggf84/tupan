#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for extensions module.
"""


import numpy as np
from pynbody.lib import extensions
from pynbody.lib.utils.timing import (Timer, timings)


def run_all():
    kernels = extensions.build_kernels()
    compare_ret_C_vs_CL(kernels)
    performance_C_vs_CL(kernels)


##############################################

#    from pynbody.io import HDF5IO
#    ic = HDF5IO("plummer0000b-3bh")
#    p = ic.read_snapshot()
#    p.set_acc(p)
#    bi = p['blackhole']
#    print(bi._pnacc)

#    pnacc = test_pnacc(kernels["c_lib64_p2p_pnacc_kernel"], bi, bi.copy())

#    print(pnacc[:,:3])

##############################################

    print(timings)



def compare_ret_C_vs_CL(kernels):
    bi = set_particles(64)
    compare(test_phi, kernels["c_lib64_p2p_phi_kernel"],
                      kernels["cl_lib64_p2p_phi_kernel"], bi)
    compare(test_acc, kernels["c_lib64_p2p_acc_kernel"],
                      kernels["cl_lib64_p2p_acc_kernel"], bi)


def performance_C_vs_CL(kernels):
    bi = set_particles(8192)
    print("\n\nDouble Precision Performance:")
    print("-----------------------------")
    performance_test(kernels["c_lib64_p2p_phi_kernel"],
                     kernels["cl_lib64_p2p_phi_kernel"], bi, output_shape="({ni},)")
    performance_test(kernels["c_lib64_p2p_acc_kernel"],
                     kernels["cl_lib64_p2p_acc_kernel"], bi, output_shape="({ni},4)")
    print("\n\nSingle Precision Performance:")
    print("-----------------------------")
    performance_test(kernels["c_lib32_p2p_phi_kernel"],
                     kernels["cl_lib32_p2p_phi_kernel"], bi, output_shape="({ni},)")
    performance_test(kernels["c_lib32_p2p_acc_kernel"],
                     kernels["cl_lib32_p2p_acc_kernel"], bi, output_shape="({ni},4)")




def set_particles(npart):
    if npart < 2: npart = 2
    from pynbody.models.imf import IMF
    from pynbody.models.plummer import Plummer
    imf = IMF.padoan2007(0.075, 120.0)
    p = Plummer(npart, imf, epsf=0.0, epstype='b', seed=1)
    p.make_plummer()
    bi = p.particles['body']
    return bi




def test_phi(cext, clext, bi, bj):
    # -------------------------------------------

    ni = len(bi)
    nj = len(bj)
    iposmass = np.vstack((bi.pos.T, bi.mass)).T
    jposmass = np.vstack((bj.pos.T, bj.mass)).T
    data = (np.uint32(ni),
            np.uint32(nj),
            iposmass, bi.eps2,
            jposmass, bj.eps2)

    output_buf = np.empty_like(bi.phi)

    # Adjusts global_size to be an integer multiple of local_size
    local_size = 384
    global_size = ((ni-1)//local_size + 1) * local_size

    # -------------------------------------------

    # XXX: kw has no efects with C extensions!
    cext.load_data(*data, global_size=global_size,
                          local_size=local_size,
                          output_buf=output_buf)
    cext.execute()
    cres = cext.get_result()
    cepot = 0.5 * np.sum(bi.mass * cres)

    # -------------------------------------------

    clext.load_data(*data, global_size=global_size,
                           local_size=local_size,
                           output_buf=output_buf)
    clext.execute()
    clres = clext.get_result()
    clepot = 0.5 * np.sum(bi.mass * clres)

    # -------------------------------------------

    return (np.allclose(cres, clres, rtol=1e-08, atol=1e-11),
            np.allclose(clres, cres, rtol=1e-08, atol=1e-11),
            cepot, clepot, np.sum(np.abs(cres - clres)))





def test_acc(cext, clext, bi, bj):
    # -------------------------------------------

    ni = len(bi)
    nj = len(bj)
    iposmass = np.vstack((bi.pos.T, bi.mass)).T
    jposmass = np.vstack((bj.pos.T, bj.mass)).T
    data = (np.uint32(ni),
            np.uint32(nj),
            iposmass, bi.eps2,
            jposmass, bj.eps2)

    output_buf = np.empty((len(bi.acc),4))

    # Adjusts global_size to be an integer multiple of local_size
    local_size = 384
    global_size = ((ni-1)//local_size + 1) * local_size

    # -------------------------------------------

    # XXX: kw has no efects with C extensions!
    cext.load_data(*data, global_size=global_size,
                          local_size=local_size,
                          output_buf=output_buf)
    cext.execute()
    cres = cext.get_result()
    c_com_force = (bi.mass * cres[:,:3].T).sum(1)
    c_com_force = np.dot(c_com_force, c_com_force)**0.5

    # -------------------------------------------

    clext.load_data(*data, global_size=global_size,
                           local_size=local_size,
                           output_buf=output_buf)
    clext.execute()
    clres = clext.get_result()
    cl_com_force = (bi.mass * clres[:,:3].T).sum(1)
    cl_com_force = np.dot(cl_com_force, cl_com_force)**0.5

    # -------------------------------------------

    return (np.allclose(cres, clres, rtol=1e-08, atol=1e-11),
            np.allclose(clres, cres, rtol=1e-08, atol=1e-11),
            c_com_force, cl_com_force, np.sum(np.abs(cres - clres)))





def compare(test, cext, clext, bi):
    npart = len(bi)
    bj = bi.copy()

    if test.__name__ is "test_phi":
        epot = bi.get_total_epot()
    if test.__name__ is "test_acc":
        com_force = (bi.mass * bi.acc.T).sum(1)
        com_force = np.dot(com_force, com_force)**0.5


    print("\nRuning {0} (varying i with j fixed):".format(test.__name__))
    for i in range(1, npart+1):
        a, b, cret, clret, cdiffcl = test(cext, clext, bi[:i], bj)
        fmt = "{0:4d} {1:4d} {2:5s} {3:5s} {4:< .16e} {5:< .16e} {6:< .16e}"
        print(fmt.format(i, npart, a, b, cret, clret, cdiffcl))
        if not a or not b:
            print('Error!!! Exiting...')
            sys.exit(0)

    print(" "*22+"-"*47)
    if test.__name__ is "test_phi":
        print(" "*34+"Total epot: {0:< .16e}".format(epot))
    if test.__name__ is "test_acc":
        print(" "*30+"Total CoMforce: {0:< .16e}".format(com_force))

    print("\nRuning {0} (varying j with i fixed):".format(test.__name__))
    for i in range(1, npart+1):
        a, b, cret, clret, cdiffcl = test(cext, clext, bi, bj[:i])
        fmt = "{0:4d} {1:4d} {2:5s} {3:5s} {4:< .16e} {5:< .16e} {6:< .16e}"
        print(fmt.format(npart, i, a, b, cret, clret, cdiffcl))
        if not a or not b:
            print('Error!!! Exiting...')
            sys.exit(0)

    print(" "*22+"-"*47)
    if test.__name__ is "test_phi":
        print(" "*34+"Total epot: {0:< .16e}".format(epot))
    if test.__name__ is "test_acc":
        print(" "*30+"Total CoMforce: {0:< .16e}".format(com_force))

    print("\nRuning {0} (varying i and j)".format(test.__name__))
    for i in range(1, npart+1):
        a, b, cret, clret, cdiffcl = test(cext, clext, bi[:i], bj[:i])
        fmt = "{0:4d} {1:4d} {2:5s} {3:5s} {4:< .16e} {5:< .16e} {6:< .16e}"
        print(fmt.format(i, i, a, b, cret, clret, cdiffcl))
        if not a or not b:
            print('Error!!! Exiting...')
            sys.exit(0)

    print(" "*22+"-"*47)
    if test.__name__ is "test_phi":
        print(" "*34+"Total epot: {0:< .16e}".format(epot))
    if test.__name__ is "test_acc":
        print(" "*30+"Total CoMforce: {0:< .16e}".format(com_force))





def performance_test(cext, clext, bi, output_shape, nsamples=5):
    timer = Timer()

    bj = bi.copy()

    # -------------------------------------------

    ni = len(bi)
    nj = len(bj)
    iposmass = np.vstack((bi.pos.T, bi.mass)).T
    jposmass = np.vstack((bj.pos.T, bj.mass)).T
    data = (np.uint32(ni),
            np.uint32(nj),
            iposmass, bi.eps2,
            jposmass, bj.eps2)

    output_buf = np.empty(eval(output_shape.format(ni=ni)))

    # Adjusts global_size to be an integer multiple of local_size
    local_size = 384
    global_size = ((ni-1)//local_size + 1) * local_size

    # -------------------------------------------

    print("\nRuning performance_test (average over {0} runs):".format(nsamples))

    # -------------------------------------------

    # XXX: kw has no efects with C extensions!
    cext.load_data(*data, global_size=global_size,
                          local_size=local_size,
                          output_buf=output_buf)
    c_elapsed = 0.0
    for i in range(nsamples):
        timer.start()
        cext.execute()
        c_elapsed += timer.elapsed()
    cres = cext.get_result()
    c_elapsed /= nsamples

    c_gflops = cext.flops_count * ni * nj * 1.0e-9
    c_metrics = {"time": c_elapsed,
                 "npart/s": np.sqrt(ni*nj)/c_elapsed,
                 "gflop/s": c_gflops/c_elapsed,
                 "src_name": cext.src_name[0]}
    print("C metrics :", c_metrics)

    # -------------------------------------------

    clext.load_data(*data, global_size=global_size,
                           local_size=local_size,
                           output_buf=output_buf)
    cl_elapsed = 0.0
    for i in range(nsamples):
        timer.start()
        clext.execute()
        cl_elapsed += timer.elapsed()
    clres = clext.get_result()
    cl_elapsed /= nsamples

    cl_gflops = clext.flops_count * ni * nj * 1.0e-9
    cl_metrics = {"time": cl_elapsed,
                  "npart/s": np.sqrt(ni*nj)/cl_elapsed,
                  "gflop/s": cl_gflops/cl_elapsed,
                  "src_name": clext.src_name[0]}
    print("CL metrics:", cl_metrics)















def test_pnacc(cext, bi, bj):
    from pynbody.lib.gravity import Clight

    clight = Clight(4, 25.0)

    # -------------------------------------------

    ni = len(bi)
    nj = len(bj)
    iposmass = np.vstack((bi.pos.T, bi.mass)).T
    jposmass = np.vstack((bj.pos.T, bj.mass)).T
    data = (np.uint32(ni),
            np.uint32(nj),
            iposmass, bi.vel,
            jposmass, bj.vel,
            np.uint32(clight.pn_order), np.float64(clight.inv1),
            np.float64(clight.inv2), np.float64(clight.inv3),
            np.float64(clight.inv4), np.float64(clight.inv5),
            np.float64(clight.inv6), np.float64(clight.inv7)
           )

    # -------------------------------------------

    # XXX: kw has no efects with C extensions!
    cext.load_data(*data)
    cext.execute()
    cres = cext.get_result()

    # -------------------------------------------

    return cres




if __name__ == "__main__":
    run_all()


########## end of file ##########
