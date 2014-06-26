# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function
import sys
import copy
import numpy as np
from ..lib import extensions as ext
from ..lib.utils.ctype import Ctype


__all__ = ['Particle', 'AbstractNbodyMethods']


class LazyProperty(object):
    def __init__(self, name, sctype, doc):
        self.name = name
        self.dtype = vars(Ctype)[sctype]
        self.__doc__ = doc

    def __get__(self, instance, cls):
        if instance is None:
            return self
        return instance._init_lazyproperty(self)


class MetaBaseNbodyMethods(type):
    def __init__(cls, *args, **kwargs):
        super(MetaBaseNbodyMethods, cls).__init__(*args, **kwargs)

        if hasattr(cls, '_init_lazyproperty'):
            setattr(cls, 'name', cls.__name__.lower())

            if hasattr(cls, 'attr_descr'):
                for name, sctype, doc in cls.attr_descr:
                    setattr(cls, name, LazyProperty(name, sctype, doc))

            if hasattr(cls, 'extra_attr_descr'):
                for name, sctype, doc in cls.extra_attr_descr:
                    setattr(cls, name, LazyProperty(name, sctype, doc))

            if hasattr(cls, 'pn_attr_descr'):
                for name, sctype, doc in cls.pn_attr_descr:
                    setattr(cls, name, LazyProperty(name, sctype, doc))

            if hasattr(cls, 'pn_extra_attr_descr'):
                for name, sctype, doc in cls.pn_extra_attr_descr:
                    setattr(cls, name, LazyProperty(name, sctype, doc))

        if hasattr(cls, 'dtype'):
            if hasattr(cls, 'attr_descr'):
                dtype = [(name, vars(Ctype)[sctype])
                         for name, sctype, doc in cls.attr_descr]
                setattr(cls, 'dtype', dtype)


BaseNbodyMethods = MetaBaseNbodyMethods('BaseNbodyMethods', (object,), {})


class NbodyMethods(BaseNbodyMethods):
    """This class holds common methods for particles in n-body systems.

    """
    # name, sctype, doc
    attr_descr = [
        ('id', 'uint', 'index'),
        ('mass', 'real', 'mass'),
        ('eps2', 'real', 'squared softening'),
        ('rx', 'real', 'x-position'),
        ('ry', 'real', 'y-position'),
        ('rz', 'real', 'z-position'),
        ('vx', 'real', 'x-velocity'),
        ('vy', 'real', 'y-velocity'),
        ('vz', 'real', 'z-velocity'),
        ('time', 'real', 'current time'),
        ('nstep', 'uint', 'step number'),
        ('tstep', 'real', 'time step'), ]

    extra_attr_descr = [
        ('phi', 'real', 'gravitational potential'),
        ('ax', 'real', 'x-acceleration'),
        ('ay', 'real', 'y-acceleration'),
        ('az', 'real', 'z-acceleration'),
        ('jx', 'real', 'x-jerk'),
        ('jy', 'real', 'y-jerk'),
        ('jz', 'real', 'z-jerk'),
        ('sx', 'real', 'x-snap'),
        ('sy', 'real', 'y-snap'),
        ('sz', 'real', 'z-snap'),
        ('cx', 'real', 'x-crackle'),
        ('cy', 'real', 'y-crackle'),
        ('cz', 'real', 'z-crackle'),
        ('tstepij', 'real', 'auxiliary time step'), ]

    def __repr__(self):
        return repr(self.__dict__)

    def copy(self):
        return copy.deepcopy(self)

    def register_attribute(self, name, sctype, doc=''):
        setattr(type(self), name, LazyProperty(name, sctype, doc))

    @property
    def pos(self):  # XXX: deprecate?
        return np.concatenate((self.rx, self.ry, self.rz,)).reshape(3, -1).T

    @property
    def vel(self):  # XXX: deprecate?
        return np.concatenate((self.vx, self.vy, self.vz,)).reshape(3, -1).T

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
        mrx = self.mass * self.rx
        mry = self.mass * self.ry
        mrz = self.mass * self.rz
        mr = np.array([mrx, mry, mrz])
        return mr.sum(1) / self.total_mass

    @property
    def com_v(self):
        """Center-of-Mass velocity of the system.

        .. note::

            Post-Newtonian corrections, if enabled, are included.

        """
        mvx = self.mass * self.vx
        mvy = self.mass * self.vy
        mvz = self.mass * self.vz
        mv = np.array([mvx, mvy, mvz])
        return mv.sum(1) / self.total_mass

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
        self.rx += com_r[0]
        self.ry += com_r[1]
        self.rz += com_r[2]
        self.vx += com_v[0]
        self.vy += com_v[1]
        self.vz += com_v[2]

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
        mvx = self.mass * self.vx
        mvy = self.mass * self.vy
        mvz = self.mass * self.vz
        return np.array([mvx, mvy, mvz]).T

    @property
    def linear_momentum(self):
        """Total linear momentum of the system.

        .. note::

            This quantity possibly includes the linear momentum of the
            center-of-mass w.r.t. the origin of coordinates.

        .. note::

            Post-Newtonian corrections, if enabled, are included.

        """
        return self.lm.sum(0)

    # -- angular momentum
    @property
    def am(self):
        """Individual angular momentum.

        .. note::

            Post-Newtonian corrections, if enabled, are included.

        """
        mvx = self.mass * self.vx
        mvy = self.mass * self.vy
        mvz = self.mass * self.vz
        amx = (self.ry * mvz) - (self.rz * mvy)
        amy = (self.rz * mvx) - (self.rx * mvz)
        amz = (self.rx * mvy) - (self.ry * mvx)
        return np.array([amx, amy, amz]).T

    @property
    def angular_momentum(self):
        """Total angular momentum of the system.

        .. note::

            This quantity possibly includes the angular momentum of the
            center-of-mass w.r.t. the origin of coordinates.

        .. note::

            Post-Newtonian corrections, if enabled, are included.

        """
        return self.am.sum(0)

    # -- kinetic energy
    @property
    def ke(self):
        """Individual kinetic energy.

        .. note::

            Post-Newtonian corrections, if enabled, are included.

        """
        ke = 0.5 * self.mass * (self.vx**2 + self.vy**2 + self.vz**2)
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

    # -- miscellaneous methods
    def min_tstep(self):
        """Minimum absolute value of tstep.

        """
        return abs(self.tstep).min()

    def max_tstep(self):
        """Maximum absolute value of tstep.

        """
        return abs(self.tstep).max()

    # -- lenght scales
    @property
    def virial_radius(self):
        """Virial radius of the system.

        """
        mtot = self.total_mass
        pe = self.potential_energy
        return (mtot**2) / (-2*pe)

    @property
    def radial_size(self):
        """Radial size of the system (a.k.a. radius of gyration).

        .. note::

            This quantity is calculated w.r.t. the center-of-mass of the
            system.

        """
        com_r = self.com_r
        rx = self.rx - com_r[0]
        ry = self.ry - com_r[1]
        rz = self.rz - com_r[2]
        mom_inertia = (self.mass * (rx**2 + ry**2 + rz**2)).sum()
        s = (mom_inertia / self.total_mass)**0.5
        return s

    # -- rescaling methods
    def dynrescale_total_mass(self, total_mass):
        """Rescales the total mass of the system while maintaining its
        dynamics unchanged.

        """
        m_ratio = total_mass / self.total_mass
        self.mass *= m_ratio
        self.rx *= m_ratio
        self.ry *= m_ratio
        self.rz *= m_ratio

    def dynrescale_radial_size(self, size):
        """Rescales the radial size of the system while maintaining its
        dynamics unchanged.

        """
        r_scale = size / self.radial_size
        v_scale = 1 / r_scale**0.5
        self.rx *= r_scale
        self.ry *= r_scale
        self.rz *= r_scale
        self.vx *= v_scale
        self.vy *= v_scale
        self.vz *= v_scale

    def dynrescale_virial_radius(self, rvir):
        """Rescales the virial radius of the system while maintaining its
        dynamics unchanged.

        """
        r_scale = rvir / self.virial_radius
        v_scale = 1 / r_scale**0.5
        self.rx *= r_scale
        self.ry *= r_scale
        self.rz *= r_scale
        self.vx *= v_scale
        self.vy *= v_scale
        self.vz *= v_scale

    def scale_to_virial(self):
        """Rescale system to virial equilibrium (2K + U = 0).

        """
        ke = self.kinetic_energy
        pe = self.potential_energy
        v_scale = ((-0.5 * pe) / ke)**0.5
        self.vx *= v_scale
        self.vy *= v_scale
        self.vz *= v_scale

    def to_nbody_units(self):
        """Rescales system to nbody units while maintaining its dynamics
        unchanged.

        """
        self.dynrescale_total_mass(1.0)
        self.dynrescale_virial_radius(1.0)


class PNbodyMethods(NbodyMethods):
    """This class holds some post-Newtonian methods.

    """
    include_pn_corrections = True

    # name, sctype, doc
    pn_attr_descr = []

    pn_extra_attr_descr = [
        ('pnax', 'real', 'PN x-acceleration'),
        ('pnay', 'real', 'PN y-acceleration'),
        ('pnaz', 'real', 'PN z-acceleration'),
        ('pn_ke', 'real', 'PN correction for kinectic energy.'),
        ('pn_mrx', 'real', 'PN correction for x-com_r'),
        ('pn_mry', 'real', 'PN correction for y-com_r'),
        ('pn_mrz', 'real', 'PN correction for z-com_r'),
        ('pn_mvx', 'real', 'PN correction for x-com_v'),
        ('pn_mvy', 'real', 'PN correction for y-com_v'),
        ('pn_mvz', 'real', 'PN correction for z-com_v'),
        ('pn_amx', 'real', 'PN correction for x-angular momentum'),
        ('pn_amy', 'real', 'PN correction for y-angular momentum'),
        ('pn_amz', 'real', 'PN correction for z-angular momentum'), ]

    @property
    def com_r(self):
        """Center-of-Mass position of the system.

        .. note::

            Post-Newtonian corrections, if enabled, are included.

        """
        rcom = super(PNbodyMethods, self).com_r
        pn_mr = np.array([self.pn_mrx, self.pn_mry, self.pn_mrz])
        pn_rcom = pn_mr.sum(1) / self.total_mass
        return rcom + pn_rcom

    @property
    def com_v(self):
        """Center-of-Mass velocity of the system.

        .. note::

            Post-Newtonian corrections, if enabled, are included.

        """
        vcom = super(PNbodyMethods, self).com_v
        pn_mv = np.array([self.pn_mvx, self.pn_mvy, self.pn_mvz])
        pn_vcom = pn_mv.sum(1) / self.total_mass
        return vcom + pn_vcom

    @property
    def lm(self):
        """Individual linear momentum.

        .. note::

            Post-Newtonian corrections, if enabled, are included.

        """
        lm = super(PNbodyMethods, self).lm
        pn_lm = np.array([self.pn_mvx, self.pn_mvy, self.pn_mvz]).T
        return lm + pn_lm

    @property
    def am(self):
        """Individual angular momentum.

        .. note::

            Post-Newtonian corrections, if enabled, are included.

        """
        am = super(PNbodyMethods, self).am
        pn_am = np.array([self.pn_amx, self.pn_amy, self.pn_amz]).T
        return am + pn_am

    @property
    def ke(self):
        """Individual kinetic energy.

        .. note::

            Post-Newtonian corrections, if enabled, are included.

        """
        ke = super(PNbodyMethods, self).ke
        pn_ke = self.pn_ke
        return ke + pn_ke

    def set_pnacc(self, other, kernel=ext.PNAcc()):
        """Set individual post-Newtonian gravitational acceleration due to
        other particles.

        """
        kernel(self, other)

    def pn_kick_ke(self, dt):
        """Kicks kinetic energy due to post-Newtonian terms.

        """
        pnfx = self.mass * self.pnax
        pnfy = self.mass * self.pnay
        pnfz = self.mass * self.pnaz
        self.pn_ke -= (self.vx * pnfx + self.vy * pnfy + self.vz * pnfz) * dt

    def pn_drift_com_r(self, dt):
        """Drifts center of mass position due to post-Newtonian terms.

        """
        self.pn_mrx += self.pn_mvx * dt
        self.pn_mry += self.pn_mvy * dt
        self.pn_mrz += self.pn_mvz * dt

    def pn_kick_lmom(self, dt):
        """Kicks linear momentum due to post-Newtonian terms.

        """
        pnfx = self.mass * self.pnax
        pnfy = self.mass * self.pnay
        pnfz = self.mass * self.pnaz
        self.pn_mvx -= pnfx * dt
        self.pn_mvy -= pnfy * dt
        self.pn_mvz -= pnfz * dt

    def pn_kick_amom(self, dt):
        """Kicks angular momentum due to post-Newtonian terms.

        """
        pnfx = self.mass * self.pnax
        pnfy = self.mass * self.pnay
        pnfz = self.mass * self.pnaz
        self.pn_amx -= (self.ry * pnfz - self.rz * pnfy) * dt
        self.pn_amy -= (self.rz * pnfx - self.rx * pnfz) * dt
        self.pn_amz -= (self.rx * pnfy - self.ry * pnfx) * dt


AbstractNbodyMethods = NbodyMethods
if '--pn_order' in sys.argv:
    AbstractNbodyMethods = PNbodyMethods


#  class Body(AbstractNbodyMethods):
#     """
#     The most basic particle type.
#     """
#     attrs = AbstractNbodyMethods.attrs + AbstractNbodyMethods.special_attrs
#     names = AbstractNbodyMethods.names + AbstractNbodyMethods.special_names
#     dtype = [(_[0], _[1]) for _ in attrs]
#     data0 = np.zeros(0, dtype)
#
#     def __init__(self, n=0, data=None):
#         """
#         Initializer
#         """
#         if data is None:
#             if n: data = np.zeros(n, self.dtype)
#             else: data = self.data0
#         self.data = data
#         self.n = len(self)
#
#     #
#     # miscellaneous methods
#     #
#
#
#     def append(self, obj):
#         if obj.n:
#             self.data = np.concatenate((self.data, obj.data))
#             self.n = len(self)
#
#
#     def remove(self, id):
#         slc = np.where(self.id == id)
#         self.data = np.delete(self.data, slc, 0)
#         self.n = len(self)
#
#
#     def insert(self, id, obj):
#         index = np.where(self.id == id)[0]
#         v = obj.data
#         self.data = np.insert(self.data, index*np.ones(len(v)), v, 0)
#         self.n = len(self)
#
#
#     def pop(self, id=None):
#         if id is None:
#             index = -1
#             id = self.id[-1]
#         else:
#             index = np.where(self.id == id)[0]
#         obj = self[index]
#         self.remove(id)
#         return obj


###############################################################################
class Particle(AbstractNbodyMethods):
    """

    """
    def __init__(self, n=0):
        self.n = n

    def update_attrs(self, attrs):
        vars(self).update(attrs)
        self.n = len(self.id)

    @classmethod
    def from_attrs(cls, attrs):
        obj = cls.__new__(cls)
        obj.update_attrs(attrs)
        return obj

    @property
    def attributes(self):
        for (k, v) in vars(self).items():
            if k not in ('n',):
                yield (k, v)

    def __str__(self):
        fmt = self.name + '(['
        if self.n:
            for (k, v) in self.attributes:
                fmt += '\n\t{0}: {1},'.format(k, v)
            fmt += '\n'
        fmt += '])'
        return fmt

    def __contains__(self, idx):
        return idx in self.id

    def __len__(self):
        return self.n

    def append(self, obj):
        if obj.n:
            attrs = ((k, np.concatenate([getattr(self, k), v]))
                     for (k, v) in obj.attributes)
            self.update_attrs(attrs)

    def __getitem__(self, index):
        if isinstance(index, int):
            attrs = ((k, v[index][None]) for (k, v) in self.attributes)
            return self.from_attrs(attrs)
        attrs = ((k, v[index]) for (k, v) in self.attributes)
        return self.from_attrs(attrs)

    def __setitem__(self, index, value):
        for (k, v) in self.attributes:
            v[index] = getattr(value, k)

    def astype(self, cls):
        newobj = cls(self.n)
        newobj.set_state(self.get_state())
        return newobj

    def get_state(self):
        array = np.zeros(self.n, dtype=self.dtype)
        for name in array.dtype.names:
            if hasattr(self, name):
                attr = getattr(self, name)
                array[name] = attr
        return array

    def set_state(self, array):
        for name in array.dtype.names:
            if hasattr(self, name):
                attr = getattr(self, name)
                attr[...] = array[name]

    def _init_lazyproperty(self, lazyprop):
        name, dtype = lazyprop.name, lazyprop.dtype
        value = np.zeros(self.n, dtype=dtype)
        setattr(self, name, value)
        return value


# -- End of File --
