#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from __future__ import print_function
import os
import sys
from warnings import warn
from functools import reduce
import numpy as np
import pyopencl as cl
from pynbody.lib.utils import (timings, Timer)


__all__ = ["Extensions"]


def get_pattern(pattern, src_code):
    for line in src_code.splitlines():
        if pattern in line:
            return line.split()[-1]



class Extensions(object):
    """

    """
    def __init__(self, path):
        self.path = path
        self.ext_type = None
        self._cl_ctx = None
        self._cl_queue = None
        self._cl_devbuf_res = None
#        print(self.path)


    # --------------------------------------------------------------------------
    # Load source

    def load_source(self, filename):
        """
        Read source files as a string
        """
        self.src_name = os.path.splitext(filename)

        # read in source code and output shape
        fname = os.path.join(self.path, filename)
        with open(fname, 'r') as fobj:
            self.src_code = fobj.read()

        # read flops count from core function
        fname = os.path.join(self.path, self.src_name[0]+"_core.h")
        with open(fname, 'r') as fobj:
            self.flops_count = int(get_pattern("Total flop count", fobj.read()))



    # --------------------------------------------------------------------------
    # Build program

    def _c_build(self, **kwargs):
        ctype = kwargs.pop("dtype", 'f')
        if not (ctype is 'd' or ctype is 'f'):
            msg = "{0}._cl_build received unexpected dtype: '{1}'."
            raise TypeError(msg.format(self.__class__.__name__, ctype))

        dummy = kwargs.pop("junroll", 1)    # Only meaningful on CL kernels.

        if kwargs:
            msg = "{0}._cl_build received unexpected keyword arguments: {1}."
            raise TypeError(msg.format(self.__class__.__name__,
                                       ", ".join(kwargs.keys())))

        cppargs = []
        signature="lib32_"+self.src_name[0]
        self.dtype = np.dtype(ctype)
        if self.dtype is np.dtype('d'):
            cppargs.append("-D DOUBLE")
            signature = signature.replace("lib32_", "lib64_")

        try:
            from instant import build_module
            prog = build_module(source_directory=self.path,
                                code=self.src_code,
                                init_code='import_array();',
                                system_headers=["numpy/arrayobject.h"],
                                include_dirs=[np.get_include(),self.path],
                                libraries=["m"],
                                cppargs=cppargs,
                                signature=signature)
            self.program = getattr(prog, self.src_name[0])
            self.ext_type = "C_EXTENSION"
        except Exception as c_build_except:
            warn("{0}".format(c_build_except), stacklevel=2)
            msg = "Sorry, I can't build C extension '{0}'.\nExiting..."
            print(msg.format(''.join(self.src_name)))
            sys.exit(0)


    def _cl_build(self, **kwargs):
        ctype = kwargs.pop("dtype", 'f')
        if not (ctype is 'd' or ctype is 'f'):
            msg = "{0}._cl_build received unexpected dtype: '{1}'."
            raise TypeError(msg.format(self.__class__.__name__, ctype))

        junroll = kwargs.pop("junroll", 1)

        if kwargs:
            msg = "{0}._cl_build received unexpected keyword arguments: {1}."
            raise TypeError(msg.format(self.__class__.__name__,
                                       ", ".join(kwargs.keys())))

        options = " -I {path}".format(path=self.path)
        options += " -D JUNROLL={junroll}".format(junroll=junroll)

        self.dtype = np.dtype(ctype)
        if self.dtype is np.dtype('d'):
            options += " -D DOUBLE"

        options += " -cl-fast-relaxed-math"

        try:
            self._cl_ctx = cl.create_some_context()
            self._cl_queue = cl.CommandQueue(self._cl_ctx)
            prog = cl.Program(self._cl_ctx, self.src_code).build(options=options)
            self.program = getattr(prog, self.src_name[0])
            self.ext_type = "CL_EXTENSION"
        except Exception as cl_build_except:
            warn("{0}".format(cl_build_except), RuntimeWarning, stacklevel=2)
            msg = "Sorry, I can't build OpenCL extension '{0}'."
            print(msg.format(''.join(self.src_name)))
            raise RuntimeWarning


    def build(self, **kwargs):
        if ".c" in self.src_name:
            self._c_build(**kwargs)
        elif ".cl" in self.src_name:
            try:
                self._cl_build(**kwargs)
            except RuntimeWarning:
                ans = raw_input("Do you want to continue with C extension ([y]/n)? ")
                ans = ans.lower()
                if ans == 'n' or ans == 'no':
                    print('Exiting...')
                    sys.exit(0)
                self.load_source(self.src_name[0]+".c")
                self._c_build(**kwargs)



    # --------------------------------------------------------------------------
    # Load data

    def _c_load_data(self, *args):
        self._kernelargs = (args,)


    def _cl_load_data(self, *args, **kwargs):
        # Get keyword arguments
        lsize = kwargs.pop("local_size")
        local_size = lsize if isinstance(lsize, tuple) else (lsize,)

        gsize = kwargs.pop("global_size")
        global_size = gsize if isinstance(gsize, tuple) else (gsize,)

        output_buf = kwargs.pop("output_buf")

        if kwargs:
            msg = "{0}._cl_load_data received unexpected keyword arguments: {1}."
            raise TypeError(msg.format(self.__class__.__name__,
                                       ", ".join(kwargs.keys())))

        # Set input buffers and kernel args on CL device
        mf = cl.mem_flags
        dev_args = [global_size, local_size]
        for item in args:
            if isinstance(item, np.ndarray):
                hostbuf = item.copy().astype(self.dtype)
                dev_args.append(cl.Buffer(self._cl_ctx,
                                          mf.READ_ONLY | mf.COPY_HOST_PTR,
                                          hostbuf=hostbuf))
            else:
                dev_args.append(item)

        # Set output buffer on CL device
        self._hostbuf_output = output_buf.copy().astype(self.dtype)
        self._cl_devbuf_res = cl.Buffer(self._cl_ctx,
                                        mf.WRITE_ONLY | mf.USE_HOST_PTR,
                                        hostbuf=self._hostbuf_output)
        dev_args.append(self._cl_devbuf_res)

        # Set local memory sizes on CL device
        base_mem_size = reduce(lambda x, y: x * y, local_size)
        base_mem_size *= self.dtype.itemsize
        local_mem_size_list = (4*base_mem_size, base_mem_size)
        for size in local_mem_size_list:
            dev_args.append(cl.LocalMemory(size))

        # Finally puts everything in _kernelargs
        self._kernelargs = dev_args


    def load_data(self, *args, **kwargs):
        if self.ext_type is "C_EXTENSION":
            self._c_load_data(*args)
        elif self.ext_type is "CL_EXTENSION":
            self._cl_load_data(*args, **kwargs)



    # --------------------------------------------------------------------------
    # Execute program

    def _c_execute(self):
        args = self._kernelargs
        self._hostbuf_output = self.program(*args)


    def _cl_execute(self):
        args = self._kernelargs
        self.program(self._cl_queue, *args).wait()


    def execute(self):
        if self.ext_type is "C_EXTENSION":
            self._c_execute()
        elif self.ext_type is "CL_EXTENSION":
            self._cl_execute()



    # --------------------------------------------------------------------------
    # Get result

    def _c_get_result(self):
        return self._hostbuf_output


    def _cl_get_result(self):
        cl.enqueue_copy(self._cl_queue, self._hostbuf_output, self._cl_devbuf_res)
        return self._hostbuf_output


    def get_result(self):
        if self.ext_type is "C_EXTENSION":
            return self._c_get_result()
        elif self.ext_type is "CL_EXTENSION":
            return self._cl_get_result()





def build_kernels():
    dirname = os.path.dirname(__file__)
    abspath = os.path.abspath(dirname)
    path = os.path.join(abspath, "ext2")

    # -------------------------------------------
    # Build kernels

    clext_phi = Extensions(path)
    clext_phi.load_source("p2p_phi_kernel.cl")
    clext_phi.build(dtype='d', junroll=16)

    # -------------------------------------------

    clext_acc = Extensions(path)
    clext_acc.load_source("p2p_acc_kernel.cl")
    clext_acc.build(dtype='d', junroll=16)

    # -------------------------------------------

    cext_phi = Extensions(path)
    cext_phi.load_source("p2p_phi_kernel.c")
    cext_phi.build(dtype='d')

    # -------------------------------------------

    cext_acc = Extensions(path)
    cext_acc.load_source("p2p_acc_kernel.c")
    cext_acc.build(dtype='d')

    # -------------------------------------------

    cext_pnacc = Extensions(path)
    cext_pnacc.load_source("p2p_pnacc_kernel.c")
    cext_pnacc.build(dtype='d')

    # -------------------------------------------

    kernels = {}
    kernels["cl_lib64_p2p_phi_kernel"] = clext_phi
    kernels["cl_lib64_p2p_acc_kernel"] = clext_acc
    kernels["c_lib64_p2p_phi_kernel"] = cext_phi
    kernels["c_lib64_p2p_acc_kernel"] = cext_acc
    kernels["c_lib64_p2p_pnacc_kernel"] = cext_pnacc

    # -------------------------------------------

    return kernels






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
                 "gflops": c_gflops/c_elapsed,
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
                  "gflops": cl_gflops/cl_elapsed,
                  "src_name": clext.src_name[0]}
    print("CL metrics:", cl_metrics)





def set_particles(npart):
    if npart < 2: npart = 2
    from pynbody.models import (IMF, Plummer)
    imf = IMF.padoan2007(0.075, 120.0)
    p = Plummer(npart, imf, epsf=0.0, epstype='b', seed=1)
    p.make_plummer()
    bi = p.particles['body']
    bj = bi.copy()
    return bi












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

    bi = set_particles(256)

    kernels = build_kernels()

    compare(test_phi, kernels["c_lib64_p2p_phi_kernel"], kernels["cl_lib64_p2p_phi_kernel"], bi)
    compare(test_acc, kernels["c_lib64_p2p_acc_kernel"], kernels["cl_lib64_p2p_acc_kernel"], bi)

    performance_test(kernels["c_lib64_p2p_phi_kernel"], kernels["cl_lib64_p2p_phi_kernel"], bi, output_shape="({ni},)")
    performance_test(kernels["c_lib64_p2p_acc_kernel"], kernels["cl_lib64_p2p_acc_kernel"], bi, output_shape="({ni},4)")


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




########## end of file ##########
