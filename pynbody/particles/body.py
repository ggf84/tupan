#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import sys
import copy
import numpy as np
from ..lib import gravity
from ..lib.utils.timing import decallmethods, timings
from ..lib.utils.memoize import cache, cache_arg


__all__ = ["Body"]


def make_attrs(cls):
    def make_property(attr, doc):
        def fget(self): return self.data[attr]
        def fset(self, value): self.data[attr] = value
        def fdel(self): raise NotImplementedError()
        return property(fget, fset, fdel, doc)
    attrs = ((i[0], cls.__name__+"\'s "+i[2]) for i in cls.attrs)
    for (attr, doc) in attrs:
        setattr(cls, attr, make_property(attr, doc))
    return cls



class NbodyUtils(object):
    """

    """
    def __len__(self):
        return len(self.id)

    def __contains__(self, id):
        return id in self.id


    def keys(self):
        return self.kind.keys()

    def objs(self):
        return self.kind.values()

    def items(self):
        return self.kind.items()

    def copy(self):
        return copy.deepcopy(self)

    def astype(self, cls):
        newobj = cls()
        for obj in self.objs():
            tmp = cls()
            tmp.set_state(obj.get_state())
            newobj.append(tmp)
        return newobj




class NbodyMethods(NbodyUtils):
    """
    This class holds common methods for particles in n-body systems.
    """
    attrs = [# name, dtype, doc
             ("id", "u8", "index"),
             ("mass", "f8", "mass"),
             ("pos", "3f8", "position"),
             ("vel", "3f8", "velocity"),
             ("eps2", "f8", "squared softening"),
             ("time", "f8", "current time"),
             ("tstep", "f8", "time step"),
             ("nstep", "u8", "step number"),
            ]
    names = [_[0] for _ in attrs]
    dtype = [(_[0], _[1]) for _ in attrs]
    data0 = np.zeros(0, dtype)


    ### total mass and center-of-mass

    @property
    @cache
    def total_mass(self):
        """
        Total mass.
        """
        return float(self.mass.sum())

    @property
    @cache
    def rcom(self):
        """
        Position of the center-of-mass.
        """
        mtot = self.total_mass
        rcom = (self.mass * self.pos.T).sum(1)
        return (rcom / mtot)

    @property
    @cache
    def vcom(self):
        """
        Velocity of the center-of-mass.
        """
        mtot = self.total_mass
        vcom = (self.mass * self.vel.T).sum(1)
        return (vcom / mtot)

    def move_to_center(self):
        """
        Moves the center-of-mass to the origin of coordinates.
        """
        self.pos -= self.rcom
        self.vel -= self.vcom


    ### linear momentum

    @property
    @cache
    def lm(self):
        """
        Individual linear momentum.
        """
        return (self.mass * self.vel.T).T

    @property
    @cache
    def linear_momentum(self):
        """
        Total linear momentum.
        """
        return self.lm.sum(0)


    ### angular momentum

    @property
    @cache
    def am(self):
        """
        Individual angular momentum.
        """
        return (self.mass * np.cross(self.pos, self.vel).T).T

    @property
    @cache
    def angular_momentum(self):
        """
        Total angular momentum.
        """
        return self.am.sum(0)


    ### kinetic energy

    @property
    @cache
    def ke(self):
        """
        Individual kinetic energy.
        """
        return 0.5 * self.mass * (self.vel**2).sum(1)

    @property
    @cache
    def kinetic_energy(self):
        """
        Total kinetic energy.
        """
        return float(self.ke.sum())


    ### potential energy

    @property
    @cache
    def pe(self):
        """
        Individual potential energy.
        """
        phi = self.get_phi(self)
        return self.mass * phi

    @property
    @cache
    def potential_energy(self):
        """
        Total potential energy.
        """
        return 0.5 * float(self.pe.sum())


    ### virial energy

    @property
    @cache
    def ve(self):
        """
        Individual virial energy.
        """
        acc = self.get_acc(self)
        force = (self.mass * acc.T).T
        return self.pos * force

    @property
    @cache
    def virial_energy(self):
        """
        Total virial energy.
        """
        return float(self.ve.sum())



    ### gravity

    @cache_arg(1)
    def get_tstep(self, objs, eta):
        """
        Get the individual time-steps due to other particles.
        """
        gravity.tstep.set_args(self, objs, eta/2)
        gravity.tstep.run()
        return gravity.tstep.get_result()

    @cache_arg(1)
    def get_phi(self, objs):
        """
        Get the individual gravitational potential due to other particles.
        """
        gravity.phi.set_args(self, objs)
        gravity.phi.run()
        return gravity.phi.get_result()

    @cache_arg(1)
    def get_acc(self, objs):
        """
        Get the individual gravitational acceleration due to other particles.
        """
        gravity.acc.set_args(self, objs)
        gravity.acc.run()
        return gravity.acc.get_result()

    @cache_arg(1)
    def get_acc_jerk(self, objs):
        """
        Get the individual gravitational acceleration and jerk due to other particles.
        """
        gravity.acc_jerk.set_args(self, objs)
        gravity.acc_jerk.run()
        return gravity.acc_jerk.get_result()


    def update_tstep(self, objs, eta):
        """
        Update the individual time-steps due to other particles.
        """
        self.tstep = self.get_tstep(objs, eta)

    def update_phi(self, objs):
        """
        Update the individual gravitational potential due to other particles.
        """
        self.phi = self.get_phi(objs)

    def update_acc(self, objs):
        """
        Update the individual gravitational acceleration due to other particles.
        """
        self.acc = self.get_acc(objs)

    def update_acc_jerk(self, objs):
        """
        Update the individual gravitational acceleration and jerk due to other particles.
        """
        (self.acc, self.jerk) = self.get_acc_jerk(objs)


    ### miscellaneous methods

    @cache
    def min_tstep(self):
        """
        Minimum absolute value of tstep.
        """
        return np.abs(self.tstep).min()

    @cache
    def max_tstep(self):
        """
        Maximum absolute value of tstep.
        """
        return np.abs(self.tstep).max()




class PNbodyMethods(NbodyMethods):
    """
    This class holds common methods for particles in n-body systems with post-Newtonian corrections.
    """
    special_attrs = [# name, dtype, doc
                     ("pnacc", "3f8", "post-Newtonian acceleration"),
                     ("ke_pn_shift", "f8", "post-Newtonian correction for the kinetic energy"),
                     ("lmom_pn_shift", "3f8", "post-Newtonian correction for the linear momentum"),
                     ("amom_pn_shift", "3f8", "post-Newtonian correction for the angular momentum"),
                     ("rcom_pn_shift", "3f8", "post-Newtonian correction for the center-of-mass position"),
                    ]
    special_names = [_[0] for _ in special_attrs]
    special_dtype = [(_[0], _[1]) for _ in special_attrs]
    special_data0 = np.zeros(0, special_dtype) if special_attrs else None

    attrs = NbodyMethods.attrs + special_attrs
    names = NbodyMethods.names + special_names
    dtype = [(_[0], _[1]) for _ in attrs]
    data0 = np.zeros(0, dtype)


    ### PN stuff

    def get_pnacc(self, objs, pn_order, clight):
        """
        Get the individual post-newtonian gravitational acceleration due to other particles.
        """
        gravity.pnacc.set_args(self, objs, pn_order, clight)
        gravity.pnacc.run()
        return gravity.pnacc.get_result()

    def update_pnacc(self, objs, pn_order, clight):
        """
        Update individual post-newtonian gravitational acceleration due to other particles.
        """
        self.pnacc = self.get_pnacc(objs, pn_order, clight)


    def evolve_ke_pn_shift(self, tstep):
        """
        Evolves kinetic energy shift in time due to post-newtonian terms.
        """
        ke_jump = tstep * (self.vel * self.pnacc).sum(1)
        self.ke_pn_shift -= ke_jump

    def get_ke_pn_shift(self):
        return (self.mass * self.ke_pn_shift).sum(0)


    def evolve_rcom_pn_shift(self, tstep):
        """
        Evolves center of mass position shift in time due to post-newtonian terms.
        """
        rcom_jump = tstep * (self.mass * self.lmom_pn_shift.T).T
        self.rcom_pn_shift += rcom_jump

    def get_rcom_pn_shift(self):
        return self.rcom_pn_shift.sum(0) / self.total_mass


    def evolve_lmom_pn_shift(self, tstep):
        """
        Evolves linear momentum shift in time due to post-newtonian terms.
        """
        lmom_jump = tstep * self.pnacc
        self.lmom_pn_shift -= lmom_jump

    def get_lmom_pn_shift(self):
        return (self.mass * self.lmom_pn_shift.T).T.sum(0)

    def get_vcom_pn_shift(self):
        return self.get_lmom_pn_shift() / self.total_mass


    def evolve_amom_pn_shift(self, tstep):
        """
        Evolves angular momentum shift in time due to post-newtonian terms.
        """
        amom_jump = tstep * np.cross(self.pos, self.pnacc)
        self.amom_pn_shift -= amom_jump

    def get_amom_pn_shift(self):
        return (self.mass * self.amom_pn_shift.T).T.sum(0)



AbstractNbodyMethods = PNbodyMethods if "--pn_order" in sys.argv else NbodyMethods


@decallmethods(timings)
@make_attrs
class Body(AbstractNbodyMethods):
    """
    The most basic particle type.
    """
    def __init__(self, n=0, data=None):
        """
        Initializer
        """
        if data is None:
            if n: data = np.zeros(n, self.dtype)
            else: data = self.data0
        self.data = data
        self.n = len(self)

    @property
    def kind(self):
        return {type(self).__name__.lower(): self}

    #
    # miscellaneous methods
    #

    def __str__(self):
        fmt = type(self).__name__+"(["
        if self.n:
            fmt += "\n"
            for obj in self:
                fmt += "{\n"
                for name in obj.data.dtype.names:
                    fmt += " {0}: {1},\n".format(name, getattr(obj, name).tolist()[0])
                fmt += "},\n"
        fmt += "])"
        return fmt


    def __repr__(self):
        fmt = type(self).__name__+"("
        if self.n: fmt += str(self.data)
        else: fmt += "[]"
        fmt += ")"
        return fmt


    def __hash__(self):
        return hash(buffer(self.data))


    def __getitem__(self, slc):
        if isinstance(slc, int):
            data = self.data[[slc]]
        if isinstance(slc, slice):
            data = self.data[slc]
        if isinstance(slc, list):
            data = self.data[slc]
        if isinstance(slc, np.ndarray):
            if slc.all():
                return self
            data = None
            if slc.any():
                data = self.data[slc]

        return type(self)(data=data)


    def append(self, obj):
        if obj.n:
            self.data = np.concatenate((self.data, obj.data))
            self.n = len(self)


    def remove(self, id):
        slc = np.where(self.id == id)
        self.data = np.delete(self.data, slc, 0)
        self.n = len(self)


    def insert(self, id, obj):
        index = np.where(self.id == id)[0]
        v = obj.data
        self.data = np.insert(self.data, index*np.ones(len(v)), v, 0)
        self.n = len(self)


    def pop(self, id=None):
        if id is None:
            index = -1
            id = self.id[-1]
        else:
            index = np.where(self.id == id)[0]
        obj = self[index]
        self.remove(id)
        return obj


    def get_state(self):
        return self.data


    def set_state(self, array):
        self.data = type(self)(len(array)).data
        self.data[:] = array
        self.n = len(self)


########## end of file ##########
