#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


import numpy as np
from .gravity import gravity_methods as gravity


__all__ = ["Interactor", "interact"]


_has_built = False
def build():
    global _has_built
    if not _has_built:
        gravity.build()
        _has_built = True


class Interactor(object):
    """

    """
    def __init__(self):
        pass


    # Potential methods

    def phi_body(self, iobj, objs):

        jobj = objs["body"]
#        if jobj:
        iphi = gravity.newtonian.set_phi(iobj, jobj)
        self_phi = iphi.copy()

        jobj = objs["blackhole"]
        if jobj:
            iphi += gravity.newtonian.set_phi(iobj, jobj)

        jobj = objs["sph"]
        if jobj:
            iphi += gravity.newtonian.set_phi(iobj, jobj)

        return (iphi, self_phi)


    def phi_blackhole(self, iobj, objs):

        jobj = objs["blackhole"]
#        if jobj:
        iphi = gravity.newtonian.set_phi(iobj, jobj)
        self_phi = iphi.copy()

        jobj = objs["body"]
        if jobj:
            iphi += gravity.newtonian.set_phi(iobj, jobj)

        jobj = objs["sph"]
        if jobj:
            iphi += gravity.newtonian.set_phi(iobj, jobj)

        return (iphi, self_phi)


    def phi_sph(self, iobj, objs):

        jobj = objs["sph"]
#        if jobj:
        iphi = gravity.newtonian.set_phi(iobj, jobj)
        self_phi = iphi.copy()

        jobj = objs["body"]
        if jobj:
            iphi += gravity.newtonian.set_phi(iobj, jobj)

        jobj = objs["blackhole"]
        if jobj:
            iphi += gravity.newtonian.set_phi(iobj, jobj)

        return (iphi, self_phi)


    # Acceleration methods

    def acc_body(self, iobj, objs):

        jobj = objs["body"]
#        if jobj:
        ret = gravity.newtonian.set_acc(iobj, jobj)
        iacc = ret[0]
        irhostep = ret[1]
        sum_nj = len(jobj)-1

        jobj = objs["blackhole"]
        if jobj:
            ret = gravity.newtonian.set_acc(iobj, jobj)
            iacc += ret[0]
            irhostep += ret[1]
            sum_nj += len(jobj)

        jobj = objs["sph"]
        if jobj:
            ret = gravity.newtonian.set_acc(iobj, jobj)
            iacc += ret[0]
            irhostep += ret[1]
            sum_nj += len(jobj)

#        return (iacc, np.sqrt(irhostep/sum_nj))
        return (iacc, irhostep)


    def acc_blackhole(self, iobj, objs):

        jobj = objs["blackhole"]
#        if jobj:
        ret = gravity.newtonian.set_acc(iobj, jobj)
        pnret = gravity.post_newtonian.set_acc(iobj, jobj)
        iacc = ret[0] + pnret[0]
        ipnacc = pnret[0]
        irhostep = ret[1]
        sum_nj = len(jobj)-1

        jobj = objs["body"]
        if jobj:
            ret = gravity.newtonian.set_acc(iobj, jobj)
            iacc += ret[0]
            irhostep += ret[1]
            sum_nj += len(jobj)

        jobj = objs["sph"]
        if jobj:
            ret = gravity.newtonian.set_acc(iobj, jobj)
            iacc += ret[0]
            irhostep += ret[1]
            sum_nj += len(jobj)

#        return (iacc, ipnacc, np.sqrt(irhostep/sum_nj))
        return (iacc, ipnacc, irhostep)


    def acc_sph(self, iobj, objs):

        jobj = objs["sph"]
#        if jobj:
        ret = gravity.newtonian.set_acc(iobj, jobj)
        iacc = ret[0]
        irhostep = ret[1]
        sum_nj = len(jobj)-1

        jobj = objs["body"]
        if jobj:
            ret = gravity.newtonian.set_acc(iobj, jobj)
            iacc += ret[0]
            irhostep += ret[1]
            sum_nj += len(jobj)

        jobj = objs["blackhole"]
        if jobj:
            ret = gravity.newtonian.set_acc(iobj, jobj)
            iacc += ret[0]
            irhostep += ret[1]
            sum_nj += len(jobj)

#        return (iacc, np.sqrt(irhostep/sum_nj))
        return (iacc, irhostep)


interact = Interactor()


########## end of file ##########
