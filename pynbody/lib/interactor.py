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

    def acc_body(self, iobj, objs):

        jobj = objs["body"]
#        if jobj:
        ret = gravitation.newtonian.set_acc(iobj, jobj)
        iacc = ret

        jobj = objs["blackhole"]
        if jobj:
            ret = gravitation.newtonian.set_acc(iobj, jobj)
            iacc += ret

        jobj = objs["sph"]
        if jobj:
            ret = gravitation.newtonian.set_acc(iobj, jobj)
            iacc += ret

        return iacc


    def acc_blackhole(self, iobj, objs):

        jobj = objs["blackhole"]
#        if jobj:
        ret = gravitation.newtonian.set_acc(iobj, jobj)
        pnret = gravitation.post_newtonian.set_acc(iobj, jobj)
        iacc = ret
        ipnacc = pnret

        jobj = objs["body"]
        if jobj:
            ret = gravitation.newtonian.set_acc(iobj, jobj)
            iacc += ret

        jobj = objs["sph"]
        if jobj:
            ret = gravitation.newtonian.set_acc(iobj, jobj)
            iacc += ret

        return iacc, ipnacc


    def acc_sph(self, iobj, objs):

        jobj = objs["sph"]
#        if jobj:
        ret = gravitation.newtonian.set_acc(iobj, jobj)
        iacc = ret

        jobj = objs["body"]
        if jobj:
            ret = gravitation.newtonian.set_acc(iobj, jobj)
            iacc += ret

        jobj = objs["blackhole"]
        if jobj:
            ret = gravitation.newtonian.set_acc(iobj, jobj)
            iacc += ret

        return iacc


    # Acceleration-Timestep methods

    def acctstep_body(self, iobj, objs, eta):

        eta34 = (3.0/4) * eta

        jobj = objs["body"]
#        if jobj:
        ret = gravitation.newtonian.set_acctstep(iobj, jobj, eta34)
        iacc = ret[0]
        iomega = ret[1]

        jobj = objs["blackhole"]
        if jobj:
            ret = gravitation.newtonian.set_acctstep(iobj, jobj, eta34)
            iacc += ret[0]
            iomega += ret[1]

        jobj = objs["sph"]
        if jobj:
            ret = gravitation.newtonian.set_acctstep(iobj, jobj, eta34)
            iacc += ret[0]
            iomega += ret[1]

        return (iacc, eta/iomega)


    def acctstep_blackhole(self, iobj, objs, eta):

        eta34 = (3.0/4) * eta

        jobj = objs["blackhole"]
#        if jobj:
        ret = gravitation.newtonian.set_acctstep(iobj, jobj, eta34)
        pnret = gravitation.post_newtonian.set_acc(iobj, jobj)
        iacc = ret[0]
        ipnacc = pnret
        iomega = ret[1]

        jobj = objs["body"]
        if jobj:
            ret = gravitation.newtonian.set_acctstep(iobj, jobj, eta34)
            iacc += ret[0]
            iomega += ret[1]

        jobj = objs["sph"]
        if jobj:
            ret = gravitation.newtonian.set_acctstep(iobj, jobj, eta34)
            iacc += ret[0]
            iomega += ret[1]

        return (iacc, ipnacc, eta/iomega)


    def acctstep_sph(self, iobj, objs, eta):

        eta34 = (3.0/4) * eta

        jobj = objs["sph"]
#        if jobj:
        ret = gravitation.newtonian.set_acctstep(iobj, jobj, eta34)
        iacc = ret[0]
        iomega = ret[1]

        jobj = objs["body"]
        if jobj:
            ret = gravitation.newtonian.set_acctstep(iobj, jobj, eta34)
            iacc += ret[0]
            iomega += ret[1]

        jobj = objs["blackhole"]
        if jobj:
            ret = gravitation.newtonian.set_acctstep(iobj, jobj, eta34)
            iacc += ret[0]
            iomega += ret[1]

        return (iacc, eta/iomega)


    # Timestep methods

    def tstep_body(self, iobj, objs, eta):

        eta34 = (3.0/4) * eta

        jobj = objs["body"]
#        if jobj:
        ret = gravitation.newtonian.set_tstep(iobj, jobj, eta34)
        iomega = ret

        jobj = objs["blackhole"]
        if jobj:
            ret = gravitation.newtonian.set_tstep(iobj, jobj, eta34)
            iomega += ret

        jobj = objs["sph"]
        if jobj:
            ret = gravitation.newtonian.set_tstep(iobj, jobj, eta34)
            iomega += ret

        return eta/iomega


    def tstep_blackhole(self, iobj, objs, eta):

        eta34 = (3.0/4) * eta

        jobj = objs["blackhole"]
#        if jobj:
        ret = gravitation.newtonian.set_tstep(iobj, jobj, eta34)
        iomega = ret

        jobj = objs["body"]
        if jobj:
            ret = gravitation.newtonian.set_tstep(iobj, jobj, eta34)
            iomega += ret

        jobj = objs["sph"]
        if jobj:
            ret = gravitation.newtonian.set_tstep(iobj, jobj, eta34)
            iomega += ret

        return eta/iomega


    def tstep_sph(self, iobj, objs, eta):

        eta34 = (3.0/4) * eta

        jobj = objs["sph"]
#        if jobj:
        ret = gravitation.newtonian.set_tstep(iobj, jobj, eta34)
        iomega = ret

        jobj = objs["body"]
        if jobj:
            ret = gravitation.newtonian.set_tstep(iobj, jobj, eta34)
            iomega += ret

        jobj = objs["blackhole"]
        if jobj:
            ret = gravitation.newtonian.set_tstep(iobj, jobj, eta34)
            iomega += ret

        return eta/iomega


interact = Interactor()


########## end of file ##########
