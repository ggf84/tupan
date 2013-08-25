# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function
import sys
import copy
import numpy as np
from ..lib import gravity
from ..lib.utils import ctype
from ..lib.utils.timing import decallmethods, timings


__all__ = ["Bodies"]


class NbodyMethods(object):
    """This class holds common methods for particles in n-body systems.

    """
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

    ### total mass and center-of-mass
    @property
    def total_mass(self):
        """Total mass.

        """
        return float(self.mass.sum())

    @property
    def rcom(self):
        """Position of the center-of-mass.

        """
        mtot = self.total_mass
        rcomx = (self.mass * self.rx).sum()
        rcomy = (self.mass * self.ry).sum()
        rcomz = (self.mass * self.rz).sum()
        return (np.array([rcomx, rcomy, rcomz]) / mtot)

    @property
    def vcom(self):
        """Velocity of the center-of-mass.

        """
        mtot = self.total_mass
        vcomx = (self.mass * self.vx).sum()
        vcomy = (self.mass * self.vy).sum()
        vcomz = (self.mass * self.vz).sum()
        return (np.array([vcomx, vcomy, vcomz]) / mtot)

    def com_to_origin(self):
        """Moves the center-of-mass to the origin of coordinates.

        """
        rcom = self.rcom
        self.rx -= rcom[0]
        self.ry -= rcom[1]
        self.rz -= rcom[2]
        vcom = self.vcom
        self.vx -= vcom[0]
        self.vy -= vcom[1]
        self.vz -= vcom[2]

    def move_com(self, rcom, vcom):
        """Moves the center-of-mass to the given coordinates.

        """
        self.com_to_origin()
        self.rx += rcom[0]
        self.ry += rcom[1]
        self.rz += rcom[2]
        self.vx += vcom[0]
        self.vy += vcom[1]
        self.vz += vcom[2]

    ### linear momentum
    @property
    def lm(self):
        """Individual linear momentum.

        """
        lmx = (self.mass * self.vx)
        lmy = (self.mass * self.vy)
        lmz = (self.mass * self.vz)
        return np.array([lmx, lmy, lmz]).T

    @property
    def linear_momentum(self):
        """Total linear momentum.

        """
        return self.lm.sum(0)

    ### angular momentum
    @property
    def am(self):
        """Individual angular momentum.

        """
        amx = self.mass * ((self.ry * self.vz) - (self.rz * self.vy))
        amy = self.mass * ((self.rz * self.vx) - (self.rx * self.vz))
        amz = self.mass * ((self.rx * self.vy) - (self.ry * self.vx))
        return np.array([amx, amy, amz]).T

    @property
    def angular_momentum(self):
        """Total angular momentum.

        """
        return self.am.sum(0)

    ### kinetic energy
    @property
    def ke(self):
        """Individual kinetic energy.

        """
        return 0.5 * self.mass * (self.vx**2 + self.vy**2 + self.vz**2)

    @property
    def kinetic_energy(self):
        """Total kinetic energy.

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
        gravity.tstep.calc(self, ps, eta)

    def set_phi(self, ps):
        """Set individual gravitational potential due to other particles.

        """
        gravity.phi.calc(self, ps)

    def set_acc(self, ps):
        """Set individual gravitational acceleration due to other particles.

        """
        gravity.acc.calc(self, ps)

    def set_acc_jerk(self, ps):
        """Set individual gravitational acceleration and jerk due to other
        particles.

        """
        gravity.acc_jerk.calc(self, ps)

    def set_snap_crackle(self, ps):
        """Set individual gravitational snap and crackle due to other
        particles.

        """
        gravity.snap_crackle.calc(self, ps)

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

        """
        rcom = self.rcom
        rx = self.rx - rcom[0]
        ry = self.ry - rcom[1]
        rz = self.rz - rcom[2]
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
    def set_pnacc(self, ps):
        """Set individual post-newtonian gravitational acceleration due to
        other particles.

        """
        gravity.pnacc.calc(self, ps)

    def pn_kick_ke(self, tau):
        """Kicks kinetic energy due to post-newtonian terms.

        """
        if not "pn_ke" in self.__dict__:
            self.register_auxiliary_attribute("pn_ke", ctype.REAL)
        self.pn_ke -= self.mass * (self.vx * self.pnax
                                   + self.vy * self.pnay
                                   + self.vz * self.pnaz) * tau

    def pn_get_ke_correction(self):
        if not "pn_ke" in self.__dict__:
            self.register_auxiliary_attribute("pn_ke", ctype.REAL)
        return float(self.pn_ke.sum())

    def pn_drift_rcom(self, tau):
        """Drifts center of mass position due to post-newtonian terms.

        """
        if not "pn_rcomx" in self.__dict__:
            self.register_auxiliary_attribute("pn_rcomx", ctype.REAL)
        if not "pn_rcomy" in self.__dict__:
            self.register_auxiliary_attribute("pn_rcomy", ctype.REAL)
        if not "pn_rcomz" in self.__dict__:
            self.register_auxiliary_attribute("pn_rcomz", ctype.REAL)
        self.pn_rcomx += self.pn_lmx * tau
        self.pn_rcomy += self.pn_lmy * tau
        self.pn_rcomz += self.pn_lmz * tau

    def pn_get_rcom_correction(self):
        if not "pn_rcomx" in self.__dict__:
            self.register_auxiliary_attribute("pn_rcomx", ctype.REAL)
        if not "pn_rcomy" in self.__dict__:
            self.register_auxiliary_attribute("pn_rcomy", ctype.REAL)
        if not "pn_rcomz" in self.__dict__:
            self.register_auxiliary_attribute("pn_rcomz", ctype.REAL)
        rcomx = self.pn_rcomx.sum()
        rcomy = self.pn_rcomy.sum()
        rcomz = self.pn_rcomz.sum()
        return np.array([rcomx, rcomy, rcomz]) / self.total_mass

    def pn_kick_lmom(self, tau):
        """Kicks linear momentum due to post-newtonian terms.

        """
        if not "pn_lmx" in self.__dict__:
            self.register_auxiliary_attribute("pn_lmx", ctype.REAL)
        if not "pn_lmy" in self.__dict__:
            self.register_auxiliary_attribute("pn_lmy", ctype.REAL)
        if not "pn_lmz" in self.__dict__:
            self.register_auxiliary_attribute("pn_lmz", ctype.REAL)
        self.pn_lmx -= self.mass * self.pnax * tau
        self.pn_lmy -= self.mass * self.pnay * tau
        self.pn_lmz -= self.mass * self.pnaz * tau

    def pn_get_lmom_correction(self):
        if not "pn_lmx" in self.__dict__:
            self.register_auxiliary_attribute("pn_lmx", ctype.REAL)
        if not "pn_lmy" in self.__dict__:
            self.register_auxiliary_attribute("pn_lmy", ctype.REAL)
        if not "pn_lmz" in self.__dict__:
            self.register_auxiliary_attribute("pn_lmz", ctype.REAL)
        lmx = self.pn_lmx.sum()
        lmy = self.pn_lmy.sum()
        lmz = self.pn_lmz.sum()
        return np.array([lmx, lmy, lmz])

    def pn_get_vcom_correction(self):
        return self.pn_get_lmom_correction() / self.total_mass

    def pn_kick_amom(self, tau):
        """Kicks angular momentum due to post-newtonian terms.

        """
        if not "pn_amx" in self.__dict__:
            self.register_auxiliary_attribute("pn_amx", ctype.REAL)
        if not "pn_amy" in self.__dict__:
            self.register_auxiliary_attribute("pn_amy", ctype.REAL)
        if not "pn_amz" in self.__dict__:
            self.register_auxiliary_attribute("pn_amz", ctype.REAL)
        self.pn_amx -= self.mass * (self.ry * self.pnaz
                                    - self.rz * self.pnay) * tau
        self.pn_amy -= self.mass * (self.rz * self.pnax
                                    - self.rx * self.pnaz) * tau
        self.pn_amz -= self.mass * (self.rx * self.pnay
                                    - self.ry * self.pnax) * tau

    def pn_get_amom_correction(self):
        if not "pn_amx" in self.__dict__:
            self.register_auxiliary_attribute("pn_amx", ctype.REAL)
        if not "pn_amy" in self.__dict__:
            self.register_auxiliary_attribute("pn_amy", ctype.REAL)
        if not "pn_amz" in self.__dict__:
            self.register_auxiliary_attribute("pn_amz", ctype.REAL)
        amx = self.pn_amx.sum()
        amy = self.pn_amy.sum()
        amz = self.pn_amz.sum()
        return np.array([amx, amy, amz])


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
