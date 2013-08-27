# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function, division
import logging
from .utils.timing import decallmethods, timings


logger = logging.getLogger(__name__)


@decallmethods(timings)
class Env(object):

    def __init__(self, backend, prec):
        self.prec = prec
        self.backend = backend
        self.fptype = "float" if prec == "single" else "double"


@decallmethods(timings)
class Module(object):

    def __init__(self, env):
        self.env = env
        if self.env.backend == "C":
            from .cffi_backend import get_lib
            self.lib = get_lib(env)
            self.ffi = self.lib._cffi_ffi
        elif self.env.backend == "CL":
            from .opencl_backend import get_lib
            self.lib = get_lib(env)
        else:
            msg = "Inappropriate 'backend': {}. Supported values: ['C', 'CL']"
            raise ValueError(msg.format(env.backend))

    def __getattr__(self, name):
        logger.debug(
            "Using '%s' from %s precision %s extension module.",
            name, self.env.prec, self.env.backend
        )
        if self.env.backend == "C":
            from .cffi_backend import CKernel
            return CKernel(self.env, self.ffi, getattr(self.lib, name))
        elif self.env.backend == "CL":
            from .opencl_backend import CLKernel
            return CLKernel(self.env, getattr(self.lib, name))


@timings
def get_kernel(name, backend, prec):
    return getattr(Module(Env(backend=backend, prec=prec)), name)


########## end of file ##########
