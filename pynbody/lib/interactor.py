#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


import numpy as np
from .gravity import gravitation


__all__ = ["Interactor", "interact"]


class Interactor(object):
    """

    """
    def __init__(self):
        pass


    # Potential methods

    def phi_body(self, iobj, objs):

        jobj = objs["body"]
#        if jobj:
        iphi = gravitation.newtonian.set_phi(iobj, jobj)
        self_phi = iphi.copy()

        jobj = objs["blackhole"]
        if jobj:
            iphi += gravitation.newtonian.set_phi(iobj, jobj)

        jobj = objs["sph"]
        if jobj:
            iphi += gravitation.newtonian.set_phi(iobj, jobj)

        return (iphi, self_phi)


    def phi_blackhole(self, iobj, objs):

        jobj = objs["blackhole"]
#        if jobj:
        iphi = gravitation.newtonian.set_phi(iobj, jobj)
        self_phi = iphi.copy()

        jobj = objs["body"]
        if jobj:
            iphi += gravitation.newtonian.set_phi(iobj, jobj)

        jobj = objs["sph"]
        if jobj:
            iphi += gravitation.newtonian.set_phi(iobj, jobj)

        return (iphi, self_phi)


    def phi_sph(self, iobj, objs):

        jobj = objs["sph"]
#        if jobj:
        iphi = gravitation.newtonian.set_phi(iobj, jobj)
        self_phi = iphi.copy()

        jobj = objs["body"]
        if jobj:
            iphi += gravitation.newtonian.set_phi(iobj, jobj)

        jobj = objs["blackhole"]
        if jobj:
            iphi += gravitation.newtonian.set_phi(iobj, jobj)

        return (iphi, self_phi)


    # Acceleration methods

    def acc_body(self, iobj, objs, tau):

        jobj = objs["body"]
#        if jobj:
        ret = gravitation.newtonian.set_acc(iobj, jobj, tau)
        iacc = ret[0]
        iomega = ret[1]

        jobj = objs["blackhole"]
        if jobj:
            ret = gravitation.newtonian.set_acc(iobj, jobj, tau)
            iacc += ret[0]
            iomega += ret[1]

        jobj = objs["sph"]
        if jobj:
            ret = gravitation.newtonian.set_acc(iobj, jobj, tau)
            iacc += ret[0]
            iomega += ret[1]

        return (iacc, iomega)


    def acc_blackhole(self, iobj, objs, tau):

        jobj = objs["blackhole"]
#        if jobj:
        ret = gravitation.newtonian.set_acc(iobj, jobj, tau)
        pnret = gravitation.post_newtonian.set_acc(iobj, jobj)
        iacc = ret[0]
        ipnacc = pnret
        iomega = ret[1]

        jobj = objs["body"]
        if jobj:
            ret = gravitation.newtonian.set_acc(iobj, jobj, tau)
            iacc += ret[0]
            iomega += ret[1]

        jobj = objs["sph"]
        if jobj:
            ret = gravitation.newtonian.set_acc(iobj, jobj, tau)
            iacc += ret[0]
            iomega += ret[1]

        return (iacc, ipnacc, iomega)


    def acc_sph(self, iobj, objs, tau):

        jobj = objs["sph"]
#        if jobj:
        ret = gravitation.newtonian.set_acc(iobj, jobj, tau)
        iacc = ret[0]
        iomega = ret[1]

        jobj = objs["body"]
        if jobj:
            ret = gravitation.newtonian.set_acc(iobj, jobj, tau)
            iacc += ret[0]
            iomega += ret[1]

        jobj = objs["blackhole"]
        if jobj:
            ret = gravitation.newtonian.set_acc(iobj, jobj, tau)
            iacc += ret[0]
            iomega += ret[1]

        return (iacc, iomega)


    # Timestep methods

    def tstep_body(self, iobj, objs, tau):

        jobj = objs["body"]
#        if jobj:
        ret = gravitation.newtonian.set_tstep(iobj, jobj, tau)
        iomega = ret

        jobj = objs["blackhole"]
        if jobj:
            ret = gravitation.newtonian.set_tstep(iobj, jobj, tau)
            iomega += ret

        jobj = objs["sph"]
        if jobj:
            ret = gravitation.newtonian.set_tstep(iobj, jobj, tau)
            iomega += ret

        return iomega


    def tstep_blackhole(self, iobj, objs, tau):

        jobj = objs["blackhole"]
#        if jobj:
        ret = gravitation.newtonian.set_tstep(iobj, jobj, tau)
        iomega = ret

        jobj = objs["body"]
        if jobj:
            ret = gravitation.newtonian.set_tstep(iobj, jobj, tau)
            iomega += ret

        jobj = objs["sph"]
        if jobj:
            ret = gravitation.newtonian.set_tstep(iobj, jobj, tau)
            iomega += ret

        return iomega


    def tstep_sph(self, iobj, objs, tau):

        jobj = objs["sph"]
#        if jobj:
        ret = gravitation.newtonian.set_tstep(iobj, jobj, tau)
        iomega = ret

        jobj = objs["body"]
        if jobj:
            ret = gravitation.newtonian.set_tstep(iobj, jobj, tau)
            iomega += ret

        jobj = objs["blackhole"]
        if jobj:
            ret = gravitation.newtonian.set_tstep(iobj, jobj, tau)
            iomega += ret

        return iomega


interact = Interactor()


########## end of file ##########
