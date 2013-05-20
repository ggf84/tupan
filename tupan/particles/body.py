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
from ..lib.utils.timing import decallmethods, timings
from ..lib.utils import ctype


__all__ = ["Bodies"]


# def make_attrs(cls):
#    def make_property(attr, doc):
#        @timings
#        def fget(self): return self.data[attr]
#        @timings
#        def fset(self, value): self.data[attr] = value
#        return property(fget=fget, fset=fset, fdel=None, doc=doc)
#    attrs = ((i[0], cls.__name__+"\'s "+i[2])
#             for i in cls.attrs+cls.special_attrs)
#    for (attr, doc) in attrs:
#        setattr(cls, attr, make_property(attr, doc))
#    return cls


class NbodyUtils(object):
    """

    """
    def __len__(self):
        return len(self.id)

    def __contains__(self, id):
        return id in self.id

    def keys(self):
        return self.kind.keys()

    def values(self):
        return self.kind.values()
    members = property(values)

    def items(self):
        return self.kind.items()

    def copy(self):
        return copy.deepcopy(self)

    def astype(self, cls):
        newobj = cls()
        for obj in self.values():
            tmp = cls(obj.n)
            tmp.set_state(obj.get_state())
            newobj.append(tmp)
        return newobj


class NbodyMethods(NbodyUtils):
    """
    This class holds common methods for particles in n-body systems.
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
        ("tstep", ctype.REAL, "time step"),
        ("nstep", ctype.UINT, "step number"),
    ]
    dtype = [(_[0], _[1]) for _ in attrs]

    special_attrs = [  # name, dtype, doc

    ]
    special_dtype = [(_[0], _[1]) for _ in special_attrs]

    @property
    def pos(self):
        return np.concatenate((self.rx, self.ry, self.rz,)).reshape(3, -1).T

#    @pos.setter
#    def pos(self, value):
#        try:
#            self.rx = value[:,0]
#            self.ry = value[:,1]
#            self.rz = value[:,2]
#        except:
#            try:
#                self.rx = value[0]
#                self.ry = value[1]
#                self.rz = value[2]
#            except:
#                self.rx = value
#                self.ry = value
#                self.rz = value

    @property
    def vel(self):
        return np.concatenate((self.vx, self.vy, self.vz,)).reshape(3, -1).T

#    @vel.setter
#    def vel(self, value):
#        try:
#            self.vx = value[:,0]
#            self.vy = value[:,1]
#            self.vz = value[:,2]
#        except:
#            try:
#                self.vx = value[0]
#                self.vy = value[1]
#                self.vz = value[2]
#            except:
#                self.vx = value
#                self.vy = value
#                self.vz = value

    ### total mass and center-of-mass
    @property
    def total_mass(self):
        """
        Total mass.
        """
        return float(self.mass.sum())

    @property
    def rcom(self):
        """
        Position of the center-of-mass.
        """
        mtot = self.total_mass
        rcomx = (self.mass * self.rx).sum()
        rcomy = (self.mass * self.ry).sum()
        rcomz = (self.mass * self.rz).sum()
        return (np.array([rcomx, rcomy, rcomz]) / mtot)

    @property
    def vcom(self):
        """
        Velocity of the center-of-mass.
        """
        mtot = self.total_mass
        vcomx = (self.mass * self.vx).sum()
        vcomy = (self.mass * self.vy).sum()
        vcomz = (self.mass * self.vz).sum()
        return (np.array([vcomx, vcomy, vcomz]) / mtot)

    def move_to_center(self):
        """
        Moves the center-of-mass to the origin of coordinates.
        """
        rcom = self.rcom
        self.rx -= rcom[0]
        self.ry -= rcom[1]
        self.rz -= rcom[2]
        vcom = self.vcom
        self.vx -= vcom[0]
        self.vy -= vcom[1]
        self.vz -= vcom[2]

    ### linear momentum
    @property
    def lm(self):
        """
        Individual linear momentum.
        """
        lmx = (self.mass * self.vx)
        lmy = (self.mass * self.vy)
        lmz = (self.mass * self.vz)
        return np.array([lmx, lmy, lmz]).T

    @property
    def linear_momentum(self):
        """
        Total linear momentum.
        """
        return self.lm.sum(0)

    ### angular momentum
    @property
    def am(self):
        """
        Individual angular momentum.
        """
        amx = self.mass * ((self.ry * self.vz) - (self.rz * self.vy))
        amy = self.mass * ((self.rz * self.vx) - (self.rx * self.vz))
        amz = self.mass * ((self.rx * self.vy) - (self.ry * self.vx))
        return np.array([amx, amy, amz]).T

    @property
    def angular_momentum(self):
        """
        Total angular momentum.
        """
        return self.am.sum(0)

    ### kinetic energy
    @property
    def ke(self):
        """
        Individual kinetic energy.
        """
        return 0.5 * self.mass * (self.vx**2 + self.vy**2 + self.vz**2)

    @property
    def kinetic_energy(self):
        """
        Total kinetic energy.
        """
        return float(self.ke.sum())

    ### potential energy
    @property
    def pe(self):
        """
        Individual potential energy.
        """
        phi, = self.get_phi(self)
        return self.mass * phi

    @property
    def potential_energy(self):
        """
        Total potential energy.
        """
        return 0.5 * float(self.pe.sum())

    ### virial energy
    @property
    def ve(self):
        """
        Individual virial energy.
        """
        ax, ay, az = self.get_acc(self)
        return self.mass * (self.rx * ax + self.ry * ay + self.rz * az)

    @property
    def virial_energy(self):
        """
        Total virial energy.
        """
        return float(self.ve.sum())

    ### gravity
    def get_tstep(self, objs, eta):
        """
        Get the individual time-steps due to other particles.
        """
        gravity.tstep.set_args(self, objs, eta)
        gravity.tstep.run()
        return gravity.tstep.get_result()

    def get_phi(self, objs):
        """
        Get the individual gravitational potential due to other particles.
        """
        gravity.phi.set_args(self, objs)
        gravity.phi.run()
        return gravity.phi.get_result()

    def get_acc(self, objs):
        """
        Get the individual gravitational acceleration due to other particles.
        """
        gravity.acc.set_args(self, objs)
        gravity.acc.run()
        return gravity.acc.get_result()

    def get_acc_jerk(self, objs):
        """
        Get the individual gravitational acceleration and jerk due to other
        particles.
        """
        gravity.acc_jerk.set_args(self, objs)
        gravity.acc_jerk.run()
        return gravity.acc_jerk.get_result()

    def update_tstep(self, objs, eta):
        """
        Update the individual time-steps due to other particles.
        """
        tstep_a, tstep_b = self.get_tstep(objs, eta)
        self.tstep = tstep_a
        return tstep_b

    def update_phi(self, objs):
        """
        Update the individual gravitational potential due to other particles.
        """
        self.phi, = self.get_phi(objs)

    def update_acc(self, objs):
        """
        Update the individual gravitational acceleration due to other
        particles.
        """
        self.ax, self.ay, self.az = self.get_acc(objs)

    def update_acc_jerk(self, objs):
        """
        Update the individual gravitational acceleration and jerk due to
        other particles.
        """
        ax, ay, az, jx, jy, jz = self.get_acc_jerk(objs)
        self.ax, self.ay, self.az = ax, ay, az
        self.jx, self.jy, self.jz = jx, jy, jz

    ### miscellaneous methods
    def min_tstep(self):
        """
        Minimum absolute value of tstep.
        """
        return np.abs(self.tstep).min()

    def max_tstep(self):
        """
        Maximum absolute value of tstep.
        """
        return np.abs(self.tstep).max()


class PNbodyMethods(NbodyMethods):
    """
    This class holds common methods for particles in n-body systems with
    post-Newtonian corrections.
    """
    special_attrs = [  # name, dtype, doc
        ("pn_dvx", ctype.REAL,
         "post-Newtonian correction for the x-velocity"),
        ("pn_dvy", ctype.REAL,
         "post-Newtonian correction for the y-velocity"),
        ("pn_dvz", ctype.REAL,
         "post-Newtonian correction for the z-velocity"),
        ("pn_ke", ctype.REAL,
         "post-Newtonian correction for the kinetic energy"),
        ("pn_rcomx", ctype.REAL,
         "post-Newtonian correction for the center-of-mass x-position"),
        ("pn_rcomy", ctype.REAL,
         "post-Newtonian correction for the center-of-mass y-position"),
        ("pn_rcomz", ctype.REAL,
         "post-Newtonian correction for the center-of-mass z-position"),
        ("pn_lmx", ctype.REAL,
         "post-Newtonian correction for the x-linear momentum"),
        ("pn_lmy", ctype.REAL,
         "post-Newtonian correction for the y-linear momentum"),
        ("pn_lmz", ctype.REAL,
         "post-Newtonian correction for the z-linear momentum"),
        ("pn_amx", ctype.REAL,
         "post-Newtonian correction for the x-angular momentum"),
        ("pn_amy", ctype.REAL,
         "post-Newtonian correction for the y-angular momentum"),
        ("pn_amz", ctype.REAL,
         "post-Newtonian correction for the z-angular momentum"),
    ]
    special_dtype = [(_[0], _[1]) for _ in special_attrs]

    ### PN stuff
    def get_pnacc(self, objs):
        """
        Get the individual post-newtonian gravitational acceleration due to
        other particles.
        """
        gravity.pnacc.set_args(self, objs)
        gravity.pnacc.run()
        return gravity.pnacc.get_result()

    def evolve_ke_pn_shift(self, tstep):
        """
        Evolves kinetic energy shift in time due to post-newtonian terms.
        """
        self.pn_ke -= self.mass * (self.vx * self.pn_dvx
                                   + self.vy * self.pn_dvy
                                   + self.vz * self.pn_dvz)

    def get_ke_pn_shift(self):
        return float(self.pn_ke.sum())

    def evolve_rcom_pn_shift(self, tstep):
        """
        Evolves center of mass position shift in time due to post-newtonian
        terms.
        """
        self.pn_rcomx += tstep * self.pn_lmx
        self.pn_rcomy += tstep * self.pn_lmy
        self.pn_rcomz += tstep * self.pn_lmz

    def get_rcom_pn_shift(self):
        rcomx = self.pn_rcomx.sum()
        rcomy = self.pn_rcomy.sum()
        rcomz = self.pn_rcomz.sum()
        return np.array([rcomx, rcomy, rcomz]) / self.total_mass

    def evolve_lmom_pn_shift(self, tstep):
        """
        Evolves linear momentum shift in time due to post-newtonian terms.
        """
        self.pn_lmx -= self.mass * self.pn_dvx
        self.pn_lmy -= self.mass * self.pn_dvy
        self.pn_lmz -= self.mass * self.pn_dvz

    def get_lmom_pn_shift(self):
        lmx = self.pn_lmx.sum()
        lmy = self.pn_lmy.sum()
        lmz = self.pn_lmz.sum()
        return np.array([lmx, lmy, lmz])

    def get_vcom_pn_shift(self):
        return self.get_lmom_pn_shift() / self.total_mass

    def evolve_amom_pn_shift(self, tstep):
        """
        Evolves angular momentum shift in time due to post-newtonian terms.
        """
        self.pn_amx -= self.mass * ((
            self.ry * self.pn_dvz) - (self.rz * self.pn_dvy))
        self.pn_amy -= self.mass * ((
            self.rz * self.pn_dvx) - (self.rx * self.pn_dvz))
        self.pn_amz -= self.mass * ((
            self.rx * self.pn_dvy) - (self.ry * self.pn_dvx))

    def get_amom_pn_shift(self):
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
#    @property
#    def kind(self):
#        return {type(self).__name__.lower(): self}
#
#    #
#    # miscellaneous methods
#    #
#
#    def __str__(self):
#        fmt = type(self).__name__+"(["
#        if self.n:
#            fmt += "\n"
#            for obj in self:
#                fmt += "{\n"
#                for name in obj.data.dtype.names:
#                    arr = getattr(obj, name)
#                    fmt += " {0}: {1},\n".format(name, arr.tolist()[0])
#                fmt += "},\n"
#        fmt += "])"
#        return fmt
#
#
#    def __repr__(self):
#        fmt = type(self).__name__+"("
#        if self.n: fmt += str(self.data)
#        else: fmt += "[]"
#        fmt += ")"
#        return fmt
#
#
#    def __hash__(self):
#        return hash(buffer(self.data))
#
#
#    def __getitem__(self, slc):
#        if isinstance(slc, int):
#            data = self.data[[slc]]
#        if isinstance(slc, slice):
#            data = self.data[slc]
#        if isinstance(slc, list):
#            data = self.data[slc]
#        if isinstance(slc, np.ndarray):
#            if slc.all():
#                return self
#            data = None
#            if slc.any():
#                data = self.data[slc]
#
#        return type(self)(data=data)
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
#        self.data[:] = array
#        self.n = len(self)


###############################################################################
@decallmethods(timings)
class Bodies(AbstractNbodyMethods):
    """

    """
    def __init__(self, n=0, items=None):
        if items is None:
            self.__dict__['id'] = np.arange(n, dtype=ctype.UINT)
            for name, dtype in self.dtype+self.special_dtype:
                self.__dict__[name] = np.zeros(n, dtype=dtype)
        else:
            self.__dict__.update(items)

    @property
    def n(self):
        return len(self)

    @property
    def kind(self):
        return {type(self).__name__.lower(): self}

    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.id)

    def __setattr__(self, name, value):
        if not name in self.__dict__:
            dtype = np.array(value).dtype
            self.__dict__[name] = np.zeros(self.n, dtype)
        self.__dict__[name][:] = value

    def __getitem__(self, slc):
        if isinstance(slc, int):
            slc = [slc]
        items = {k: v[slc] for k, v in self.__dict__.items()}
        return type(self)(items=items)

    def copy(self):
        return copy.deepcopy(self)

    def append(self, obj):
        if obj.n:
            items = {k: np.concatenate((self.__dict__.get(k, []), v))
                     for k, v in obj.__dict__.items()}
            self.__dict__.update(items)

    def get_state(self):
        array = np.zeros(self.n, dtype=self.dtype)
        for name in array.dtype.names:
            array[name] = getattr(self, name)
        return array

    def set_state(self, array):
        for name in array.dtype.names:
            if name in self.__dict__:
                setattr(self, name, array[name])


########## end of file ##########
