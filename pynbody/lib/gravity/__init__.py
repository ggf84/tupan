#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A minimal module for gravity methods.
"""


from .newtonian import *
from .post_newtonian import *


__all__ = ['newtonian', 'post_newtonian']


HAS_BUILT = False

def build_kernels():
    global HAS_BUILT
    if not HAS_BUILT:
        try:
            from pynbody.lib.clkernels import (cl_p2p_acc, cl_p2p_phi)
            if not cl_p2p_acc.has_built():
                cl_p2p_acc.build_kernel()
            if not cl_p2p_phi.has_built():
                cl_p2p_phi.build_kernel()
#            raise
        except Exception as e:
            cl_p2p_acc.is_available = False
            cl_p2p_phi.is_available = False
            print(e)
            ans = raw_input("A problem occurred with the loading of the OpenCL "
                            "kernels.\nAttempting to continue with C extensions "
                            "on the CPU only.\nDo you want to continue ([y]/n)? ")
            ans = ans.lower()
            if ans == 'n' or ans == 'no':
                import sys
                print('exiting...')
                sys.exit(0)
        HAS_BUILT = True



########## end of file ##########
