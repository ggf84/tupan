#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from __future__ import print_function
import sys
import copy
import numpy as np
from .sph import Sph
from .body import Body
from .blackhole import BlackHole
from .pbase import AbstractNbodyMethods
from ..lib import gravity
from ..lib.utils.timing import decallmethods, timings


__all__ = ["Particles"]


ALL_PARTICLE_TYPES = ["sph", "body", "blackhole"]


def make_common_attrs(cls):
    def make_property(attr, doc):
        def fget(self):
            seq = [getattr(obj, attr) for obj in self.values() if obj.n]
            if len(seq) == 1:
                return seq[0]
            if len(seq) > 1:
                return np.concatenate(seq)
            return np.concatenate([getattr(obj, attr) for obj in self.values()])
        def fset(self, value):
            for obj in self.values():
                if obj.n:
                    try:
                        setattr(obj, attr, value[:obj.n])
                        value = value[obj.n:]
                    except:
                        setattr(obj, attr, value)
        def fdel(self):
            raise NotImplementedError()
        return property(fget, fset, fdel, doc)
    attrs = ((i[0], cls.__name__+'\'s '+i[2]) for i in cls.common_attributes)
    for (attr, doc) in attrs:
        setattr(cls, attr, make_property(attr, doc))
    return cls


@decallmethods(timings)
@make_common_attrs
class Particles(dict, AbstractNbodyMethods):
    """
    This class holds the particle types in the simulation.
    """
    def __init__(self, nstar=0, nbh=0, nsph=0):
        """
        Initializer
        """
        super(Particles, self).__init__(
                                        sph=Sph(nsph),
                                        body=Body(nstar),
                                        blackhole=BlackHole(nbh),
                                       )
        self.__dict__.update(self)


    #
    # miscellaneous methods
    #

    def __str__(self):
        fmt = self.__class__.__name__+'{'
        for key, obj in self.items():
            fmt += '\n{0},'.format(obj)
        fmt += '\n}'
        return fmt


    def __hash__(self):
        i = None
        for obj in self.values():
            if i is None:
                i = hash(obj)
            else:
                i ^= hash(obj)
        return i


    def __len__(self):
        return sum([obj.n for obj in self.values()])
    n = property(__len__)


    @property
    def empty(self):
        return self.__class__()


    def copy(self):
        return copy.deepcopy(self)


    def append(self, objs):
        if objs.n:
            if isinstance(objs, Particles):
                for (key, obj) in objs.items():
                    if obj.n:
                        self[key].append(obj)
            else:
                key = objs.__class__.__name__.lower()
                self[key].append(objs)


    def select(self, function):
        subset = self.empty
        for (key, obj) in self.items():
            if obj.n:
                subset[key].append(obj.select(function))
        return subset


    #
    # uncommon methods
    #

    ### total mass and center-of-mass

    def get_rcom_pn_shift(self):
        """

        """
        mtot = self.get_total_mass()
        rcom_shift = 0.0
        for obj in self.values():
            if obj.n:
                if hasattr(obj, "get_pn_correction_for_center_of_mass_position"):
                    rcom_shift += obj.get_pn_correction_for_center_of_mass_position()
        return (rcom_shift / mtot)

    def get_vcom_pn_shift(self):
        """

        """
        mtot = self.get_total_mass()
        vcom_shift = 0.0
        for obj in self.values():
            if obj.n:
                if hasattr(obj, "get_pn_correction_for_center_of_mass_velocity"):
                    vcom_shift += obj.get_pn_correction_for_center_of_mass_velocity()
        return (vcom_shift / mtot)


    ### linear momentum

    def get_lmom_pn_shift(self):
        """

        """
        lmom_shift = 0.0
        for obj in self.values():
            if obj.n:
                if hasattr(obj, "get_pn_correction_for_total_linear_momentum"):
                    lmom_shift += obj.get_pn_correction_for_total_linear_momentum()
        return lmom_shift


    ### angular momentum

    def get_amom_pn_shift(self):
        """

        """
        amom_shift = 0.0
        for obj in self.values():
            if obj.n:
                if hasattr(obj, "get_pn_correction_for_total_angular_momentum"):
                    amom_shift += obj.get_pn_correction_for_total_angular_momentum()
        return amom_shift


    ### kinetic energy

    def get_ke_pn_shift(self):
        """

        """
        ke_shift = 0.0
        for obj in self.values():
            if obj.n:
                if hasattr(obj, "get_pn_correction_for_total_energy"):
                    ke_shift += obj.get_pn_correction_for_total_energy()
        return ke_shift


    ### gravity

    def update_acc_jerk(self, objs):
        """
        Update the individual gravitational acceleration and jerk due to other particles.
        """
        gravity.acc_jerk.set_args(self, objs)
        gravity.acc_jerk.run()
        (self.acc, jerk) = gravity.acc_jerk.get_result()
        for iobj in self.values():
            if iobj.n:
                iobj.jerk = jerk[:iobj.n]
                jerk = jerk[iobj.n:]

    def update_pnacc(self, objs, pn_order, clight):
        """
        Update the individual post-newtonian gravitational acceleration due to other particles.
        """
        ni = self.blackhole.n
        nj = objs.blackhole.n
        if ni and nj:
            gravity.pnacc.set_args(self.blackhole, objs.blackhole, pn_order, clight)
            gravity.pnacc.run()
            result = gravity.pnacc.get_result()
            for iobj in self.values():
                if iobj.n:
                    if hasattr(iobj, "pnacc"):
                        iobj.pnacc = result[:iobj.n]
                        result = result[iobj.n:]


    ### prev/next timestep

    def power_averaged_dt_prev(self, power):
        """

        """
        n = 0
        average = 0.0
        for obj in self.values():
            if obj.n:
                average += (obj.dt_prev**power).sum()
                n += obj.n
        average = (average / n)**(1.0/power)
        return average


    def power_averaged_dt_next(self, power):
        """

        """
        n = 0
        average = 0.0
        for obj in self.values():
            if obj.n:
                average += (obj.dt_next**power).sum()
                n += obj.n
        average = (average / n)**(1.0/power)
        return average


########## end of file ##########
