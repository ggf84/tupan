# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function
import sys
import copy
import numpy as np
from ..lib import extensions
from ..lib.utils.timing import timings, bind_all
from ..lib.utils.ctype import Ctype


__all__ = ["Bodies"]


def typed_property(name, expected_type,
                   doc=None, can_get=True,
                   can_set=True, can_del=False):
    storage_name = '_' + name

    def fget(self):
        value = getattr(self, storage_name, None)
        if value is None:
            def get_value(self):
                if hasattr(self, 'members'):
                    arrays = [getattr(member, name)
                              for member in self.members.values()]
                    return np.concatenate(arrays)
                return expected_type(self)
            value = get_value(self)
            setattr(self, name, value)
            return value
        return value

    def fset(self, value):
        if hasattr(self, 'members'):
            if not hasattr(self, storage_name):
                ns = 0
                nf = 0
                for member in self.members.values():
                    nf += member.n
                    setattr(member, name, value[ns:nf])
                    ns += member.n
        setattr(self, storage_name, value)

    def fdel(self):
        if hasattr(self, storage_name):
            delattr(self, storage_name)
        if hasattr(self, 'members'):
            for member in self.members.values():
                delattr(member, name)

    fget.__name__ = name
    fset.__name__ = name
    fdel.__name__ = name
    return property(fget if can_get else None,
                    fset if can_set else None,
                    fdel if can_del else None,
                    doc)


class NbodyMethods(object):
    """This class holds common methods for particles in n-body systems.

    """
    id = typed_property('id',
                        lambda self: np.arange(self.n, dtype=Ctype.uint))
    mass = typed_property('mass',
                          lambda self: np.zeros(self.n, dtype=Ctype.real))
    eps2 = typed_property('eps2',
                          lambda self: np.zeros(self.n, dtype=Ctype.real))
    rx = typed_property('rx',
                        lambda self: np.zeros(self.n, dtype=Ctype.real))
    ry = typed_property('ry',
                        lambda self: np.zeros(self.n, dtype=Ctype.real))
    rz = typed_property('rz',
                        lambda self: np.zeros(self.n, dtype=Ctype.real))
    vx = typed_property('vx',
                        lambda self: np.zeros(self.n, dtype=Ctype.real))
    vy = typed_property('vy',
                        lambda self: np.zeros(self.n, dtype=Ctype.real))
    vz = typed_property('vz',
                        lambda self: np.zeros(self.n, dtype=Ctype.real))
    time = typed_property('time',
                          lambda self: np.zeros(self.n, dtype=Ctype.real))
    nstep = typed_property('nstep',
                           lambda self: np.zeros(self.n, dtype=Ctype.uint))
    tstep = typed_property('tstep',
                           lambda self: np.zeros(self.n, dtype=Ctype.real))

    include_pn_corrections = False

    attrs = [  # name, sctype, doc
        ("id", 'uint', "index"),
        ("mass", 'real', "mass"),
        ("eps2", 'real', "squared softening"),
        ("rx", 'real', "x-position"),
        ("ry", 'real', "y-position"),
        ("rz", 'real', "z-position"),
        ("vx", 'real', "x-velocity"),
        ("vy", 'real', "y-velocity"),
        ("vz", 'real', "z-velocity"),
        ("time", 'real', "current time"),
        ("nstep", 'uint', "step number"),
        ("tstep", 'real', "time step"),
    ]

    def copy(self):
        return copy.deepcopy(self)

    def register_attribute(self, attr, sctype, doc=''):
        dtype = vars(Ctype)[sctype]
        setattr(type(self), attr,
                typed_property(attr,
                               lambda x: np.zeros(x.n, dtype=dtype),
                               doc=doc, can_del=True))

    @property       # TODO: @classproperty ???
    def dtype(self):
        return [(name, vars(Ctype)[sctype])
                for name, sctype, _ in self.attrs]

    @property
    def pos(self):  # XXX: deprecate?
        return np.concatenate((self.rx, self.ry, self.rz,)).reshape(3, -1).T

    @property
    def vel(self):  # XXX: deprecate?
        return np.concatenate((self.vx, self.vy, self.vz,)).reshape(3, -1).T

    @property
    def px(self):
        return self.mass * self.vx

    @property
    def py(self):
        return self.mass * self.vy

    @property
    def pz(self):
        return self.mass * self.vz

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
        mr = np.array([mrx, mry, mrz]).T
        return mr.sum(0) / self.total_mass

    @property
    def com_v(self):
        """Center-of-Mass velocity of the system.

        .. note::

            Post-Newtonian corrections, if enabled, are included.

        """
        mvx = self.mass * self.vx
        mvy = self.mass * self.vy
        mvz = self.mass * self.vz
        mv = np.array([mvx, mvy, mvz]).T
        return mv.sum(0) / self.total_mass

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
    def set_tstep(self, ps, eta):
        """Set individual time-steps due to other particles.

        """
        extensions.tstep(self, ps, eta=eta)

    def set_phi(self, ps):
        """Set individual gravitational potential due to other particles.

        """
        extensions.phi(self, ps)

    def set_acc(self, ps):
        """Set individual gravitational acceleration due to other particles.

        """
        extensions.acc(self, ps)

    def set_pnacc(self, ps):
        """Set individual post-Newtonian gravitational acceleration due to
        other particles.

        """
        extensions.pnacc(self, ps)

    def set_acc_jerk(self, ps):
        """Set individual gravitational acceleration and jerk due to other
        particles.

        """
        extensions.acc_jerk(self, ps)

    def set_snap_crackle(self, ps):
        """Set individual gravitational snap and crackle due to other
        particles.

        """
        extensions.snap_crackle(self, ps)

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

    # -- PN stuff
    # -- TODO: move these methods to a more appropriate place...

    pn_ke = typed_property('pn_ke',
                           lambda self: np.zeros(self.n, dtype=Ctype.real))
    pn_mrx = typed_property('pn_mrx',
                            lambda self: np.zeros(self.n, dtype=Ctype.real))
    pn_mry = typed_property('pn_mry',
                            lambda self: np.zeros(self.n, dtype=Ctype.real))
    pn_mrz = typed_property('pn_mrz',
                            lambda self: np.zeros(self.n, dtype=Ctype.real))
    pn_mvx = typed_property('pn_mvx',
                            lambda self: np.zeros(self.n, dtype=Ctype.real))
    pn_mvy = typed_property('pn_mvy',
                            lambda self: np.zeros(self.n, dtype=Ctype.real))
    pn_mvz = typed_property('pn_mvz',
                            lambda self: np.zeros(self.n, dtype=Ctype.real))
    pn_amx = typed_property('pn_amx',
                            lambda self: np.zeros(self.n, dtype=Ctype.real))
    pn_amy = typed_property('pn_amy',
                            lambda self: np.zeros(self.n, dtype=Ctype.real))
    pn_amz = typed_property('pn_amz',
                            lambda self: np.zeros(self.n, dtype=Ctype.real))

    pn_attrs = [  # name, sctype, doc
        ("pn_ke", 'real', "post-newtonian correction for kinectic energy."),
        ("pn_mrx", 'real', "post-newtonian correction for x-com_r"),
        ("pn_mry", 'real', "post-newtonian correction for y-com_r"),
        ("pn_mrz", 'real', "post-newtonian correction for z-com_r"),
        ("pn_mvx", 'real', "post-newtonian correction for x-com_v"),
        ("pn_mvy", 'real', "post-newtonian correction for y-com_v"),
        ("pn_mvz", 'real', "post-newtonian correction for z-com_v"),
        ("pn_amx", 'real', "post-newtonian correction for x-angular momentum"),
        ("pn_amy", 'real', "post-newtonian correction for y-angular momentum"),
        ("pn_amz", 'real', "post-newtonian correction for z-angular momentum"),
    ]

    @property
    def com_r(self):
        """Center-of-Mass position of the system.

        .. note::

            Post-Newtonian corrections, if enabled, are included.

        """
        rcom = super(PNbodyMethods, self).com_r
        pn_mr = np.array([self.pn_mrx, self.pn_mry, self.pn_mrz]).T
        pn_rcom = pn_mr.sum(0) / self.total_mass
        return rcom + pn_rcom

    @property
    def com_v(self):
        """Center-of-Mass velocity of the system.

        .. note::

            Post-Newtonian corrections, if enabled, are included.

        """
        vcom = super(PNbodyMethods, self).com_v
        pn_mv = np.array([self.pn_mvx, self.pn_mvy, self.pn_mvz]).T
        pn_vcom = pn_mv.sum(0) / self.total_mass
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
if "--pn_order" in sys.argv:
    AbstractNbodyMethods = PNbodyMethods


# @bind_all(timings)
# @make_attrs
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
@bind_all(timings)
class Bodies(AbstractNbodyMethods):
    """

    """
    def __init__(self, n=0, items=None):
        if items is None:
            self.n = n

            # allocate attrs for the first time.
            attrs = self.attrs[:]
            if hasattr(self, 'pn_attrs'):
                attrs += self.pn_attrs
            for (attr, _, _) in attrs:
                getattr(self, attr)
        else:
            self.__dict__.update(items)
            self.n = len(self.id)

    @property
    def attributes(self):
        return ((k, v) for (k, v) in self.__dict__.items()
                if k not in ('n',))

    def __repr__(self):
        return repr(dict(self.attributes))

    def __str__(self):
        fmt = type(self).__name__+"(["
        if self.n:
            for (k, v) in self.attributes:
                fmt += "\n\t{0}: {1},".format(k, v)
            fmt += "\n"
        fmt += "])"
        return fmt

    def __contains__(self, idx):
        return idx in self.id

    def __len__(self):
        return self.n

    def append(self, obj):
        if obj.n:
            items = {k: np.concatenate((getattr(self, k), v))
                     for (k, v) in obj.attributes}
            self.__dict__.update(items)
            self.n = len(self.id)

    def __getitem__(self, slc):
        idx = [slc] if isinstance(slc, int) else slc
        items = {k: v[idx] for (k, v) in self.attributes}
        return type(self)(items=items)

    def __setitem__(self, slc, values):
        for (k, v) in self.attributes:
            v[slc] = getattr(values, k)

    def astype(self, cls):
        newobj = cls(self.n)
        newobj.set_state(self.get_state())
        return newobj

    def get_state(self):
        array = np.zeros(self.n, dtype=self.dtype)
        for name in array.dtype.names:
            if hasattr(self, name):
                array[name] = getattr(self, name)
        return array

    def set_state(self, array):
        for name in array.dtype.names:
            if hasattr(self, name):
                setattr(self, name, array[name])


# -- End of File --
