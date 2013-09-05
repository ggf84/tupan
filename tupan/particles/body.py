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
from ..lib.utils import ctype
from ..lib.utils.timing import decallmethods, timings


__all__ = ["Bodies"]


class NbodyMethods(object):
    """This class holds common methods for particles in n-body systems.

    """
    include_pn_corrections = False

    attrs = [  # name, dtype, doc
        ("id", ctype.UINT, "index"),
        ("mass", ctype.REAL, "mass"),
        ("eps2", ctype.REAL, "squared softening"),
        ("rx", ctype.REAL, "x-position"),
        ("ry", ctype.REAL, "y-position"),
        ("rz", ctype.REAL, "z-position"),
        ("vx", ctype.REAL, "x-velocity"),
        ("vy", ctype.REAL, "y-velocity"),
        ("vz", ctype.REAL, "z-velocity"),
        ("time", ctype.REAL, "current time"),
        ("nstep", ctype.UINT, "step number"),
        ("tstep", ctype.REAL, "time step"),
    ]
    dtype = [(_[0], _[1]) for _ in attrs]

    special_attrs = [  # name, dtype, doc

    ]
    special_dtype = [(_[0], _[1]) for _ in special_attrs]

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

    ### total mass and center-of-mass methods
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
        if self.include_pn_corrections:
            if not "pn_mrx" in self.__dict__:
                self.register_auxiliary_attribute("pn_mrx", ctype.REAL)
            if not "pn_mry" in self.__dict__:
                self.register_auxiliary_attribute("pn_mry", ctype.REAL)
            if not "pn_mrz" in self.__dict__:
                self.register_auxiliary_attribute("pn_mrz", ctype.REAL)
            mrx += self.pn_mrx
            mry += self.pn_mry
            mrz += self.pn_mrz
        mr = np.array([mrx, mry, mrz]).T
        return mr.sum(0) / self.total_mass

    @property
    def com_v(self):
        """Center-of-Mass velocity of the system.

        .. note::

            Post-Newtonian corrections, if enabled, are included.

        """
        mvx, mvy, mvz = self.px, self.py, self.pz
        if self.include_pn_corrections:
            if not "pn_mvx" in self.__dict__:
                self.register_auxiliary_attribute("pn_mvx", ctype.REAL)
            if not "pn_mvy" in self.__dict__:
                self.register_auxiliary_attribute("pn_mvy", ctype.REAL)
            if not "pn_mvz" in self.__dict__:
                self.register_auxiliary_attribute("pn_mvz", ctype.REAL)
            mvx += self.pn_mvx
            mvy += self.pn_mvy
            mvz += self.pn_mvz
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

    def com_to_origin(self):
        """Moves the center-of-mass to the origin of coordinates.

        """
        com_r = self.com_r
        self.rx -= com_r[0]
        self.ry -= com_r[1]
        self.rz -= com_r[2]
        com_v = self.com_v
        self.vx -= com_v[0]
        self.vy -= com_v[1]
        self.vz -= com_v[2]

    def com_move_to(self, com_r, com_v):
        """Moves the center-of-mass to the given coordinates.

        """
        self.com_to_origin()
        self.rx += com_r[0]
        self.ry += com_r[1]
        self.rz += com_r[2]
        self.vx += com_v[0]
        self.vy += com_v[1]
        self.vz += com_v[2]

    ### linear momentum
    @property
    def lm(self):
        """Individual linear momentum.

        .. note::

            Post-Newtonian corrections, if enabled, are included.

        """
        lmx, lmy, lmz = self.px, self.py, self.pz
        if self.include_pn_corrections:
            if not "pn_mvx" in self.__dict__:
                self.register_auxiliary_attribute("pn_mvx", ctype.REAL)
            if not "pn_mvy" in self.__dict__:
                self.register_auxiliary_attribute("pn_mvy", ctype.REAL)
            if not "pn_mvz" in self.__dict__:
                self.register_auxiliary_attribute("pn_mvz", ctype.REAL)
            lmx += self.pn_mvx
            lmy += self.pn_mvy
            lmz += self.pn_mvz
        return np.array([lmx, lmy, lmz]).T

    @property
    def linear_momentum(self):
        """Total linear momentum of the system.

        .. note::

            This quantity includes a possible linear momentum of the center-of-mass,
            w.r.t. the origin of coordinates.

            Post-Newtonian corrections, if enabled, are included.

        """
        return self.lm.sum(0)

    ### angular momentum
    @property
    def am(self):
        """Individual angular momentum.

        .. note::

            Post-Newtonian corrections, if enabled, are included.

        """
        px, py, pz = self.px, self.py, self.pz
        amx = (self.ry * pz) - (self.rz * py)
        amy = (self.rz * px) - (self.rx * pz)
        amz = (self.rx * py) - (self.ry * px)
        if self.include_pn_corrections:
            if not "pn_amx" in self.__dict__:
                self.register_auxiliary_attribute("pn_amx", ctype.REAL)
            if not "pn_amy" in self.__dict__:
                self.register_auxiliary_attribute("pn_amy", ctype.REAL)
            if not "pn_amz" in self.__dict__:
                self.register_auxiliary_attribute("pn_amz", ctype.REAL)
            amx += self.pn_amx
            amy += self.pn_amy
            amz += self.pn_amz
        return np.array([amx, amy, amz]).T


    @property
    def angular_momentum(self):
        """Total angular momentum of the system.

        .. note::

            This quantity includes a possible angular momentum of the center-of-mass,
            w.r.t. the origin of coordinates.

            Post-Newtonian corrections, if enabled, are included.

        """
        return self.am.sum(0)

    ### kinetic energy
    @property
    def ke(self):
        """Individual kinetic energy.

        .. note::

            Post-Newtonian corrections, if enabled, are included.

        """
        ke = 0.5 * self.mass * (self.vx**2 + self.vy**2 + self.vz**2)
        if self.include_pn_corrections:
            if not "pn_ke" in self.__dict__:
                self.register_auxiliary_attribute("pn_ke", ctype.REAL)
            ke += self.pn_ke
        return ke

    @property
    def kinetic_energy(self):
        """Total kinetic energy of the system.

        .. note::

            This quantity includes a possible kinetic energy of the center-of-mass,
            w.r.t. the origin of coordinates.

            Post-Newtonian corrections, if enabled, are included.

        """
        return float(self.ke.sum())

    ### potential energy
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

    ### virial energy
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

    ### gravity
    def set_tstep(self, ps, eta):
        """Set individual time-steps due to other particles.

        """
        extensions.tstep.calc(self, ps, eta)

    def set_phi(self, ps):
        """Set individual gravitational potential due to other particles.

        """
        extensions.phi.calc(self, ps)

    def set_acc(self, ps):
        """Set individual gravitational acceleration due to other particles.

        """
        extensions.acc.calc(self, ps)

    def set_pnacc(self, ps):
        """Set individual post-Newtonian gravitational acceleration due to
        other particles.

        """
        extensions.pnacc.calc(self, ps)

    def set_acc_jerk(self, ps):
        """Set individual gravitational acceleration and jerk due to other
        particles.

        """
        extensions.acc_jerk.calc(self, ps)

    def set_snap_crackle(self, ps):
        """Set individual gravitational snap and crackle due to other
        particles.

        """
        extensions.snap_crackle.calc(self, ps)

    ### miscellaneous methods
    def min_tstep(self):
        """Minimum absolute value of tstep.

        """
        return abs(self.tstep).min()

    def max_tstep(self):
        """Maximum absolute value of tstep.

        """
        return abs(self.tstep).max()

    ### lenght scales
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

            This quantity is calculated w.r.t. the center-of-mass of the system.

        """
        com_r = self.com_r
        rx = self.rx - com_r[0]
        ry = self.ry - com_r[1]
        rz = self.rz - com_r[2]
        I = (self.mass * (rx**2 + ry**2 + rz**2)).sum()
        s = (I / self.total_mass)**0.5
        return s

    ### rescaling methods
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
    ### PN stuff
    ### TODO: move these methods to a more appropriate place...

    def pn_kick_ke(self, tau):
        """Kicks kinetic energy due to post-Newtonian terms.

        """
        if not "pn_ke" in self.__dict__:
            self.register_auxiliary_attribute("pn_ke", ctype.REAL)
        pnfx = self.mass * self.pnax
        pnfy = self.mass * self.pnay
        pnfz = self.mass * self.pnaz
        self.pn_ke -= (self.vx * pnfx + self.vy * pnfy + self.vz * pnfz) * tau

    def pn_drift_com_r(self, tau):
        """Drifts center of mass position due to post-Newtonian terms.

        """
        if not "pn_mrx" in self.__dict__:
            self.register_auxiliary_attribute("pn_mrx", ctype.REAL)
        if not "pn_mry" in self.__dict__:
            self.register_auxiliary_attribute("pn_mry", ctype.REAL)
        if not "pn_mrz" in self.__dict__:
            self.register_auxiliary_attribute("pn_mrz", ctype.REAL)
        self.pn_mrx += self.pn_mvx * tau
        self.pn_mry += self.pn_mvy * tau
        self.pn_mrz += self.pn_mvz * tau

    def pn_kick_lmom(self, tau):
        """Kicks linear momentum due to post-Newtonian terms.

        """
        if not "pn_mvx" in self.__dict__:
            self.register_auxiliary_attribute("pn_mvx", ctype.REAL)
        if not "pn_mvy" in self.__dict__:
            self.register_auxiliary_attribute("pn_mvy", ctype.REAL)
        if not "pn_mvz" in self.__dict__:
            self.register_auxiliary_attribute("pn_mvz", ctype.REAL)
        pnfx = self.mass * self.pnax
        pnfy = self.mass * self.pnay
        pnfz = self.mass * self.pnaz
        self.pn_mvx -= pnfx * tau
        self.pn_mvy -= pnfy * tau
        self.pn_mvz -= pnfz * tau

    def pn_kick_amom(self, tau):
        """Kicks angular momentum due to post-Newtonian terms.

        """
        if not "pn_amx" in self.__dict__:
            self.register_auxiliary_attribute("pn_amx", ctype.REAL)
        if not "pn_amy" in self.__dict__:
            self.register_auxiliary_attribute("pn_amy", ctype.REAL)
        if not "pn_amz" in self.__dict__:
            self.register_auxiliary_attribute("pn_amz", ctype.REAL)
        pnfx = self.mass * self.pnax
        pnfy = self.mass * self.pnay
        pnfz = self.mass * self.pnaz
        self.pn_amx -= (self.ry * pnfz - self.rz * pnfy) * tau
        self.pn_amy -= (self.rz * pnfx - self.rx * pnfz) * tau
        self.pn_amz -= (self.rx * pnfy - self.ry * pnfx) * tau


AbstractNbodyMethods = NbodyMethods
if "--pn_order" in sys.argv:
    AbstractNbodyMethods = PNbodyMethods


#@decallmethods(timings)
#@make_attrs
# class Body(AbstractNbodyMethods):
#    """
#    The most basic particle type.
#    """
#    attrs = AbstractNbodyMethods.attrs + AbstractNbodyMethods.special_attrs
#    names = AbstractNbodyMethods.names + AbstractNbodyMethods.special_names
#    dtype = [(_[0], _[1]) for _ in attrs]
#    data0 = np.zeros(0, dtype)
#
#    def __init__(self, n=0, data=None):
#        """
#        Initializer
#        """
#        if data is None:
#            if n: data = np.zeros(n, self.dtype)
#            else: data = self.data0
#        self.data = data
#        self.n = len(self)
#
#    #
#    # miscellaneous methods
#    #
#
#
#    def append(self, obj):
#        if obj.n:
#            self.data = np.concatenate((self.data, obj.data))
#            self.n = len(self)
#
#
#    def remove(self, id):
#        slc = np.where(self.id == id)
#        self.data = np.delete(self.data, slc, 0)
#        self.n = len(self)
#
#
#    def insert(self, id, obj):
#        index = np.where(self.id == id)[0]
#        v = obj.data
#        self.data = np.insert(self.data, index*np.ones(len(v)), v, 0)
#        self.n = len(self)
#
#
#    def pop(self, id=None):
#        if id is None:
#            index = -1
#            id = self.id[-1]
#        else:
#            index = np.where(self.id == id)[0]
#        obj = self[index]
#        self.remove(id)
#        return obj
#
#
#    def get_state(self):
#        return self.data
#
#
#    def set_state(self, array):
#        self.data[...] = array
#        self.n = len(self)


###############################################################################
@decallmethods(timings)
class Bodies(AbstractNbodyMethods):
    """

    """
    def __init__(self, n=0, items=None):
        if items is None:
            for (name, dtype) in self.dtype+self.special_dtype:
                self.__dict__[name] = np.zeros(n, dtype=dtype)
            self.id[...] = np.arange(n, dtype=ctype.UINT)
        else:
            self.__dict__.update(items)

    def __repr__(self):
        return repr(self.__dict__)

    def __str__(self):
        fmt = type(self).__name__+"(["
        if self.n:
            for (k, v) in self.__dict__.items():
                fmt += "\n\t{0}: {1},".format(k, v)
            fmt += "\n"
        fmt += "])"
        return fmt

    def __contains__(self, id):
        return id in self.id

    def __len__(self):
        return len(self.id)

    @property
    def n(self):
        return len(self)

    def copy(self):
        return copy.deepcopy(self)

    def append(self, obj):
        if obj.n:
            items = {k: np.concatenate((getattr(self, k), v))
                     for (k, v) in obj.__dict__.items()}
            self.__dict__.update(items)

    def __getitem__(self, slc):
        if isinstance(slc, int):
            slc = [slc]
        items = {k: v[slc] for (k, v) in self.__dict__.items()}
        return type(self)(items=items)

    def __setitem__(self, slc, values):
        for (k, v) in self.__dict__.items():
            v[slc] = getattr(values, k)

    def astype(self, cls):
        newobj = cls()
        tmp = cls(self.n)
        tmp.set_state(self.get_state())
        newobj.append(tmp)
        return newobj

    def get_state(self):
        array = np.zeros(self.n, dtype=self.dtype)
        for name in array.dtype.names:
            array[name] = getattr(self, name)
        return array

    def set_state(self, array):
        for name in array.dtype.names:
            if name in self.__dict__:
                self.__dict__[name][...] = array[name]


########## end of file ##########
