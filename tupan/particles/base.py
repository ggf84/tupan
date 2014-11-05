# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function
import abc
import copy
import numpy as np
from ..lib import extensions as ext
from ..lib.utils import with_metaclass
from ..lib.utils.ctype import Ctype


__all__ = ['AbstractParticle', 'AbstractNbodyMethods']


class MetaParticle(abc.ABCMeta):
    def __init__(cls, *args, **kwargs):
        super(MetaParticle, cls).__init__(*args, **kwargs)

        if hasattr(cls, 'name'):
            setattr(cls, 'name', cls.__name__.lower())

            if hasattr(cls, 'default_attr_descr'):
                dtype = [(name, vars(Ctype)[sctype], shape)
                         for name, shape, sctype, _ in cls.default_attr_descr]
                setattr(cls, 'dtype', np.dtype(dtype))

            attr_descrs = []
            if hasattr(cls, 'default_attr_descr'):
                attr_descrs += cls.default_attr_descr
            if hasattr(cls, 'extra_attr_descr'):
                attr_descrs += cls.extra_attr_descr
            if hasattr(cls, 'pn_default_attr_descr'):
                attr_descrs += cls.pn_default_attr_descr
            if hasattr(cls, 'pn_extra_attr_descr'):
                attr_descrs += cls.pn_extra_attr_descr

            attr_names = [name for name, _, _, _ in attr_descrs]
            setattr(cls, 'attr_names', attr_names)
            attr_descrs = {name: (shape, sctype, doc)
                           for name, shape, sctype, doc in attr_descrs}
            setattr(cls, 'attr_descrs', attr_descrs)


class AbstractParticle(with_metaclass(MetaParticle, object)):
    """

    """
    def __init__(self, n=0):
        self.n = n

    def update_attrs(self, attrs):
        vars(self).update(attrs)
        self.n = len(self.pid)

    @classmethod
    def from_attrs(cls, attrs):
        obj = cls.__new__(cls)
        obj.update_attrs(attrs)
        return obj

    def register_attribute(self, name, shape, sctype, doc=''):
        if name not in self.attr_names:
            self.attr_names.append(name)
            self.attr_descrs[name] = (shape, sctype, doc)

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return repr(vars(self))

    def __str__(self):
        fmt = self.name + '(['
        if self.n:
            for name in self.attr_names:
                ary = getattr(self, name)
                fmt += '\n\t{0}: {1},'.format(name, ary)
            fmt += '\n'
        fmt += '])'
        return fmt

    def __contains__(self, idx):
        return idx in self.pid

    def __len__(self):
        return self.n

    def append(self, other):
        if other.n:
            attrs = []
            for name in self.attr_names:
                sary = getattr(self, name)
                oary = getattr(other, name)
                cary = np.concatenate([sary, oary], 1 if sary.ndim > 1 else 0)
                attrs.append((name, cary))
            self.update_attrs(attrs)

    def __getitem__(self, index):
        idx = (Ellipsis, index)
        if isinstance(index, int):
            idx += (None,)
        attrs = [(name, getattr(self, name)[idx])
                 for name in self.attr_names]
        return self.from_attrs(attrs)

    def __setitem__(self, index, value):
        idx = (Ellipsis, index)
        for name in self.attr_names:
            ary = getattr(self, name)
            ary[idx] = getattr(value, name)

    def __getattr__(self, name):
        if name not in self.attr_names:
            raise AttributeError(name)
        shape, sctype, _ = self.attr_descrs[name]
        value = np.zeros(shape + (self.n,), dtype=vars(Ctype)[sctype])
        setattr(self, name, value)
        return value

    def astype(self, cls):
        newobj = cls(self.n)
        newobj.set_state(self.get_state())
        return newobj

    def get_state(self):
        array = np.zeros(self.n, dtype=self.dtype)
        for name in array.dtype.names:
            if hasattr(self, name):
                attr = getattr(self, name)
                array[name] = attr.T
        return array

    def set_state(self, array):
        for name in array.dtype.names:
            if hasattr(self, name):
                attr = getattr(self, name)
                attr[...] = array[name].T


###############################################################################


class AbstractNbodyMethods(with_metaclass(abc.ABCMeta, object)):
    """This class holds common methods for particles in n-body systems.

    """
    include_pn_corrections = False

    # name, shape, sctype, doc
    default_attr_descr = [
        ('pid', (), 'uint', 'particle id'),
        ('mass', (), 'real', 'mass'),
        ('pos', (3,), 'real', 'position'),
        ('vel', (3,), 'real', 'velocity'),
        ('eps2', (), 'real', 'squared softening'),
        ('time', (), 'real', 'current time'),
        ('nstep', (), 'uint', 'step number'),
        ('tstep', (), 'real', 'time step'),
    ]

    extra_attr_descr = [
        ('phi', (), 'real', 'gravitational potential'),
        ('acc', (3,), 'real', 'acceleration'),
        ('jrk', (3,), 'real', 'jerk'),
        ('snp', (3,), 'real', 'snap'),
        ('crk', (3,), 'real', 'crackle'),
        ('tstepij', (), 'real', 'auxiliary time step'),
    ]

    pn_default_attr_descr = []

    pn_extra_attr_descr = [
        ('pnacc', (3,), 'real', 'PN acceleration'),
        ('pn_mr', (3,), 'real', 'PN correction for com_r'),
        ('pn_mv', (3,), 'real', 'PN correction for com_v'),
        ('pn_am', (3,), 'real', 'PN correction for angular momentum'),
        ('pn_ke', (), 'real', 'PN correction for kinectic energy.'),
    ]

    # -- total mass and center-of-mass methods
    @property
    def total_mass(self):
        """Total mass of the system.

        """
        return float(self.mass.sum())

    @property
    def com_r(self):
        """Center-of-Mass position of the system.

        .. note::

            Post-Newtonian corrections, if enabled, are included.

        """
        mr = self.mass * self.pos
        rcom = mr.sum(1) / self.total_mass
        if self.include_pn_corrections:
            rcom += self.pn_mr.sum(1) / self.total_mass
        return rcom

    @property
    def com_v(self):
        """Center-of-Mass velocity of the system.

        .. note::

            Post-Newtonian corrections, if enabled, are included.

        """
        mv = self.mass * self.vel
        vcom = mv.sum(1) / self.total_mass
        if self.include_pn_corrections:
            vcom += self.pn_mv.sum(1) / self.total_mass
        return vcom

    @property
    def com_linear_momentum(self):
        """Center-of-Mass linear momentum of the system.

        """
        mtot = self.total_mass
        com_v = self.com_v
        return mtot * com_v

    @property
    def com_angular_momentum(self):
        """Center-of-Mass angular momentum of the system.

        """
        mtot = self.total_mass
        com_r = self.com_r
        com_v = self.com_v
        return mtot * np.cross(com_r, com_v)

    @property
    def com_kinetic_energy(self):
        """Center-of-Mass kinetic energy of the system.

        """
        mtot = self.total_mass
        com_v = self.com_v
        return 0.5 * mtot * (com_v**2).sum()

    def com_move_to(self, com_r, com_v):
        """Moves the center-of-mass to the given coordinates.

        """
        self.pos.T[...] += com_r
        self.vel.T[...] += com_v

    def com_to_origin(self):
        """Moves the center-of-mass to the origin of coordinates.

        """
        self.com_move_to(-self.com_r, -self.com_v)

    # -- linear momentum
    @property
    def lm(self):
        """Individual linear momentum.

        .. note::

            Post-Newtonian corrections, if enabled, are included.

        """
        mv = self.mass * self.vel
        if self.include_pn_corrections:
            mv += self.pn_mv
        return mv

    @property
    def linear_momentum(self):
        """Total linear momentum of the system.

        .. note::

            This quantity possibly includes the linear momentum of the
            center-of-mass w.r.t. the origin of coordinates.

        .. note::

            Post-Newtonian corrections, if enabled, are included.

        """
        return self.lm.sum(1)

    # -- angular momentum
    @property
    def am(self):
        """Individual angular momentum.

        .. note::

            Post-Newtonian corrections, if enabled, are included.

        """
        mv = self.mass * self.vel
        am = np.cross(self.pos.T, mv.T).T
        if self.include_pn_corrections:
            am += self.pn_am
        return am

    @property
    def angular_momentum(self):
        """Total angular momentum of the system.

        .. note::

            This quantity possibly includes the angular momentum of the
            center-of-mass w.r.t. the origin of coordinates.

        .. note::

            Post-Newtonian corrections, if enabled, are included.

        """
        return self.am.sum(1)

    # -- kinetic energy
    @property
    def ke(self):
        """Individual kinetic energy.

        .. note::

            Post-Newtonian corrections, if enabled, are included.

        """
        ke = 0.5 * self.mass * (self.vel**2).sum(0)
        if self.include_pn_corrections:
            ke += self.pn_ke
        return ke

    @property
    def kinetic_energy(self):
        """Total kinetic energy of the system.

        .. note::

            This quantity possibly includes the kinetic energy of the
            center-of-mass w.r.t. the origin of coordinates.

        .. note::

            Post-Newtonian corrections, if enabled, are included.

        """
        return float(self.ke.sum())

    # -- potential energy
    @property
    def pe(self):
        """Individual potential energy.

        """
        self.set_phi(self)
        return self.mass * self.phi

    @property
    def potential_energy(self):
        """Total potential energy.

        """
        return 0.5 * float(self.pe.sum())

    # -- virial energy
    @property
    def ve(self):
        """Individual virial energy.

        """
        return 2 * self.ke + self.pe

    @property
    def virial_energy(self):
        """Total virial energy.

        """
        return 2 * self.kinetic_energy + self.potential_energy

    # -- gravity
    def set_tstep(self, other, eta, kernel=ext.Tstep()):
        """Set individual time-steps due to other particles.

        """
        kernel(self, other, eta=eta)

    def set_phi(self, other, kernel=ext.Phi()):
        """Set individual gravitational potential due to other particles.

        """
        kernel(self, other)

    def set_acc(self, other, kernel=ext.Acc()):
        """Set individual gravitational acceleration due to other particles.

        """
        kernel(self, other)

    def set_pnacc(self, other, kernel=ext.PNAcc()):
        """Set individual post-Newtonian gravitational acceleration due to
        other particles.

        """
        kernel(self, other)

    def set_acc_jerk(self, other, kernel=ext.AccJerk()):
        """Set individual gravitational acceleration and jerk due to other
        particles.

        """
        kernel(self, other)

    def set_snap_crackle(self, other, kernel=ext.SnapCrackle()):
        """Set individual gravitational snap and crackle due to other
        particles.

        """
        kernel(self, other)

    # -- lenght scales
    @property
    def virial_radius(self):
        """Virial radius of the system.

        """
        mtot = self.total_mass
        pe = self.potential_energy
        return (mtot**2) / (-2 * pe)

    @property
    def radial_size(self):
        """Radial size of the system (a.k.a. radius of gyration).

        .. note::

            This quantity is calculated w.r.t. the center-of-mass of the
            system.

        """
        com_r = self.com_r
        pos = (self.pos.T - com_r).T
        mr2 = (self.mass * pos**2).sum()
        r2 = mr2 / self.total_mass
        return r2**0.5

    # -- rescaling methods
    def dynrescale_total_mass(self, total_mass):
        """Rescales the total mass of the system while maintaining its
        dynamics unchanged.

        """
        m_ratio = total_mass / self.total_mass
        self.mass *= m_ratio
        self.pos *= m_ratio

    def dynrescale_radial_size(self, size):
        """Rescales the radial size of the system while maintaining its
        dynamics unchanged.

        """
        r_scale = size / self.radial_size
        v_scale = 1 / r_scale**0.5
        self.pos *= r_scale
        self.vel *= v_scale

    def dynrescale_virial_radius(self, rvir):
        """Rescales the virial radius of the system while maintaining its
        dynamics unchanged.

        """
        r_scale = rvir / self.virial_radius
        v_scale = 1 / r_scale**0.5
        self.pos *= r_scale
        self.vel *= v_scale

    def scale_to_virial(self):
        """Rescale system to virial equilibrium (2K + U = 0).

        """
        ke = self.kinetic_energy
        pe = self.potential_energy
        v_scale = ((-0.5 * pe) / ke)**0.5
        self.vel *= v_scale

    def to_nbody_units(self):
        """Rescales system to nbody units while maintaining its dynamics
        unchanged.

        """
        self.dynrescale_total_mass(1.0)
        self.dynrescale_virial_radius(1.0)


# -- End of File --
