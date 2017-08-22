# -*- coding: utf-8 -*-
#

"""
TODO.
"""

from __future__ import print_function
import copy
import numpy as np
from .body import Body
from .sph import Sph
from .star import Star
from .blackhole import Blackhole
from .base import MetaParticle, AbstractNbodyMethods
from ..lib import extensions as ext
from ..lib.utils import with_metaclass


class Members(dict):
    def __init__(self, **members):
        super(Members, self).__init__(**members)
        self.__dict__ = self


class ParticleSystem(with_metaclass(MetaParticle, AbstractNbodyMethods)):
    """
    This class holds the particle types in the simulation.
    """
    name = None

    def __init__(self, nbody=0, nstar=0, nbh=0, nsph=0):
        """
        Initializer.
        """
        members = {cls.name: cls(n) for (n, cls) in [(nbody, Body),
                                                     (nstar, Star),
                                                     (nbh, Blackhole),
                                                     (nsph, Sph)] if n}
        self.set_members(**members)
        if self.n:
            self.reset_pid()

    def reset_pid(self):
        for member in self.members.values():
            member.pid[...] = range(member.n)

    def set_members(self, **members):
        self.members = Members(**members)
        self.n = len(self)
        for member in self.members.values():
            self.attr_names.update(**member.attr_names)

    def update_members(self, **members):
        vars(self).clear()
        self.set_members(**members)

    @classmethod
    def from_members(cls, **members):
        obj = cls.__new__(cls)
        obj.set_members(**members)
        return obj

    def register_attribute(self, name, shape, sctype, doc=''):
        for member in self.members.values():
            member.register_attribute(name, shape, sctype, doc)

        if name not in self.attr_names:
            self.attr_names[name] = (shape, sctype, doc)

    def astype(self, name='body'):
        cls = globals()[name.capitalize()]
        obj = cls()
        for p in self.members.values():
            if p.n:
                obj += p.astype(cls)
        return self.from_members(**{cls.name: obj})

    #
    # miscellaneous methods
    #

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return repr(vars(self))

    def __str__(self):
        fmt = self.name + '(['
        if self.n:
            for member in self.members.values():
                fmt += '\n\t{0},'.format('\n\t'.join(str(member).split('\n')))
            fmt += '\n'
        fmt += '])'
        return fmt

    def __len__(self):
        return sum(len(member) for member in self.members.values())

    def __add__(self, other):
        if other.n:
            members = self.members
            for name, member in other.members.items():
                try:
                    members[name] += member
                except KeyError:
                    members[name] = member.copy()
            self.update_members(**members)
        return self

    __radd__ = __add__

    def split_by(self, mask_function):
        d_a, d_b = {}, {}
        for name, member in self.members.items():
            d_a[name], d_b[name] = member.split_by(mask_function)
        return (self.from_members(**d_a),
                self.from_members(**d_b))

    def __getitem__(self, index):
        if isinstance(index, np.ndarray):
            ns, nf = 0, 0
            members = {}
            for name, member in self.members.items():
                nf += member.n
                members[name] = member[index[ns:nf]]
                ns += member.n
            return self.from_members(**members)

        if isinstance(index, int):
            if abs(index) > self.n-1:
                msg = 'index {0} out of bounds 0<=index<{1}'
                raise IndexError(msg.format(index, self.n))
            if index < 0:
                index = self.n + index
            members = {}
            for name, member in self.members.items():
                if 0 <= index < member.n:
                    members[name] = member[index]
                index -= member.n
            return self.from_members(**members)

        if isinstance(index, slice):
            start = index.start
            stop = index.stop
            if start is None:
                start = 0
            if stop is None:
                stop = self.n
            if start < 0:
                start = self.n + start
            if stop < 0:
                stop = self.n + stop
            members = {}
            for name, member in self.members.items():
                if stop >= 0 and start < member.n:
                    members[name] = member[start-member.n:stop]
                start -= member.n
                stop -= member.n
            return self.from_members(**members)

    def __setitem__(self, index, value):
        if isinstance(index, np.ndarray):
            ns, nf = 0, 0
            for name, member in self.members.items():
                nf += member.n
                if name in value.members:
                    member[index[ns:nf]] = value.members[name]
                ns += member.n
            return

        if isinstance(index, int):
            if abs(index) > self.n-1:
                msg = 'index {0} out of bounds 0<=index<{1}'
                raise IndexError(msg.format(index, self.n))
            if index < 0:
                index = self.n + index
            for name, member in self.members.items():
                if 0 <= index < member.n:
                    if name in value.members:
                        member[index] = value.members[name]
                index -= member.n
            return

        if isinstance(index, slice):
            start = index.start
            stop = index.stop
            if start is None:
                start = 0
            if stop is None:
                stop = self.n
            if start < 0:
                start = self.n + start
            if stop < 0:
                stop = self.n + stop
            for name, member in self.members.items():
                if stop >= 0 and start < member.n:
                    if name in value.members:
                        member[start-member.n:stop] = value.members[name]
                start -= member.n
                stop -= member.n
            return

    def __getattr__(self, name):
        if name not in self.attr_names:
            raise AttributeError(name)
        members = self.members
        if len(members) == 1:
            member = next(iter(members.values()))
            value = getattr(member, name)
            setattr(self, name, value)
            return value
        arrays = [getattr(member, name) for member in members.values()
                  if name in member.attr_names]
        value = np.concatenate(arrays, -1)  # along last dimension
        ns, nf = 0, 0
        for member in members.values():
            if name in member.attr_names:
                nf += member.n
                v = value[..., ns:nf]
#                setattr(member, name, v)
                setattr(member, name, np.array(v, copy=False, order='C'))
                ns += member.n
        setattr(self, name, value)
        return value

    @property
    def global_time(self):
        return min(abs(p.time).min() for p in self.members.values() if p.n)

    @property
    def tstep_min(self):
        return min(abs(p.tstep).min() for p in self.members.values() if p.n)

    #
    # n-body methods
    #

    # -- total mass and center-of-mass
    @property
    def total_mass(self):
        """Total mass of the system.

        """
        return float(sum(p.mass.sum() for p in self.members.values() if p.n))

    @property
    def com_r(self):
        """Center-of-Mass position of the system.

        .. note::

            Post-Newtonian corrections, if enabled, are included.

        """
        mr_sum = 0
        for p in self.members.values():
            if p.n:
                mr = p.mass * p.rdot[0]
                if hasattr(p, 'pn_mr'):
                    mr += p.pn_mr
                mr_sum += mr.sum(1)
        return mr_sum / self.total_mass

    @property
    def com_v(self):
        """Center-of-Mass velocity of the system.

        .. note::

            Post-Newtonian corrections, if enabled, are included.

        """
        mv_sum = 0
        for p in self.members.values():
            if p.n:
                mv = p.mass * p.rdot[1]
                if hasattr(p, 'pn_mv'):
                    mv += p.pn_mv
                mv_sum += mv.sum(1)
        return mv_sum / self.total_mass

    @property
    def com_linear_momentum(self):
        """Center-of-Mass linear momentum of the system.

        """
        return self.total_mass * self.com_v

    @property
    def com_angular_momentum(self):
        """Center-of-Mass angular momentum of the system.

        """
        return self.total_mass * np.cross(self.com_r, self.com_v)

    @property
    def com_kinetic_energy(self):
        """Center-of-Mass kinetic energy of the system.

        """
        return 0.5 * self.total_mass * (self.com_v**2).sum()

    def com_move_to(self, com_r, com_v):
        """Moves the center-of-mass to the given coordinates.

        """
        for p in self.members.values():
            if p.n:
                p.rdot[0].T[...] += com_r
                p.rdot[1].T[...] += com_v

    def com_to_origin(self):
        """Moves the center-of-mass to the origin of coordinates.

        """
        self.com_move_to(-self.com_r, -self.com_v)

    # -- linear and angular momentum
    @property
    def linear_momentum(self):
        """Total linear momentum of the system.

        .. note::

            Post-Newtonian corrections, if enabled, are included.

        """
        lm_sum = 0
        for p in self.members.values():
            if p.n:
                lm = p.mass * p.rdot[1]
                if hasattr(p, 'pn_mv'):
                    lm += p.pn_mv
                lm_sum += lm.sum(1)
        return lm_sum

    @property
    def angular_momentum(self):
        """Total angular momentum of the system.

        .. note::

            Post-Newtonian corrections, if enabled, are included.

        """
        am_sum = 0
        for p in self.members.values():
            if p.n:
                mv = p.mass * p.rdot[1]
                am = np.cross(p.rdot[0].T, mv.T).T
                if hasattr(p, 'pn_am'):
                    am += p.pn_am
                am_sum += am.sum(1)
        return am_sum

    # -- kinetic and potential energy
    @property
    def kinetic_energy(self):
        """Total kinetic energy of the system.

        .. note::

            Post-Newtonian corrections, if enabled, are included.

        """
        ke_sum = 0
        for p in self.members.values():
            if p.n:
                v2 = (p.rdot[1]**2).sum(0)
                ke = 0.5 * p.mass * v2
                if hasattr(p, 'pn_ke'):
                    ke += p.pn_ke
                ke_sum += ke.sum()
        return float(ke_sum)

    @property
    def potential_energy(self):
        """Total potential energy.

        """
        pe_sum = 0
        for p in self.members.values():
            if p.n:
                pe = p.mass * p.phi
                pe_sum += pe.sum()
        return 0.5 * float(pe_sum)

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
        """Radial size (a.k.a. radius of gyration) w.r.t. the
        center-of-mass of the system.

        """
        mr2_sum = 0
        com_r = self.com_r
        for p in self.members.values():
            if p.n:
                pos = (p.rdot[0].T - com_r).T
                mr2 = p.mass * pos**2
                mr2_sum += mr2.sum()
        return (mr2_sum / self.total_mass)**0.5

    # -- rescaling methods
    def dynrescale_total_mass(self, total_mass):
        """Rescales the total mass of the system while maintaining its
        dynamics unchanged.

        """
        m_ratio = total_mass / self.total_mass
        for p in self.members.values():
            if p.n:
                p.mass *= m_ratio
                p.rdot[0] *= m_ratio

    def dynrescale_radial_size(self, size):
        """Rescales the radial size of the system while maintaining its
        dynamics unchanged.

        """
        r_scale = size / self.radial_size
        v_scale = 1 / r_scale**0.5
        for p in self.members.values():
            if p.n:
                p.rdot[0] *= r_scale
                p.rdot[1] *= v_scale

    def dynrescale_virial_radius(self, rvir):
        """Rescales the virial radius of the system while maintaining its
        dynamics unchanged.

        """
        self.set_phi(self)
        r_scale = rvir / self.virial_radius
        v_scale = 1 / r_scale**0.5
        for p in self.members.values():
            if p.n:
                p.rdot[0] *= r_scale
                p.rdot[1] *= v_scale

    def scale_to_virial(self):
        """Rescale system to virial equilibrium (2K + U = 0).

        """
        self.set_phi(self)
        ke = self.kinetic_energy
        pe = self.potential_energy
        v_scale = ((-0.5 * pe) / ke)**0.5
        for p in self.members.values():
            if p.n:
                p.rdot[1] *= v_scale

    def to_nbody_units(self):
        """Rescales system to nbody units while maintaining its dynamics
        unchanged.

        """
        self.dynrescale_total_mass(1.0)
        self.dynrescale_virial_radius(1.0)

    # -- O(N^2) methods
    def set_tstep(self, other, eta,
                  kernel=ext.get_kernel('Tstep')):
        """Set individual time-steps due to other particles.

        """
        i_tstep = {i: 0 for i in self.members.keys()}
        j_tstep = {j: 0 for j in other.members.keys()}
        i_tstep_sum = {i: 0 for i in self.members.keys()}
        j_tstep_sum = {j: 0 for j in other.members.keys()}

        for i, ip in self.members.items():
            if ip.n:
                for j, jp in other.members.items():
                    if jp.n:
                        kernel(ip, jp, eta=eta)
                        i_tstep_sum[i] += ip.tstep_sum
                        j_tstep_sum[j] += jp.tstep_sum
                        i_tstep[i] = np.maximum(i_tstep[i], ip.tstep)
                        j_tstep[j] = np.maximum(j_tstep[j], jp.tstep)

        if self != other:
            for j, jp in other.members.items():
                if jp.n:
                    jp.tstep[...] = eta / np.sqrt(j_tstep[j])
                    jp.tstep_sum[...] = eta / np.sqrt(j_tstep_sum[j])
        for i, ip in self.members.items():
            if ip.n:
                ip.tstep[...] = eta / np.sqrt(i_tstep[i])
                ip.tstep_sum[...] = eta / np.sqrt(i_tstep_sum[i])

    def set_phi(self, other,
                kernel=ext.get_kernel('Phi')):
        """Set individual gravitational potential due to other particles.

        """
        i_phi = {i: 0 for i in self.members.keys()}
        j_phi = {j: 0 for j in other.members.keys()}

        for i, ip in self.members.items():
            if ip.n:
                for j, jp in other.members.items():
                    if jp.n:
                        kernel(ip, jp)
                        i_phi[i] += ip.phi
                        j_phi[j] += jp.phi

        if self != other:
            for j, jp in other.members.items():
                if jp.n:
                    jp.phi[...] = j_phi[j]
        for i, ip in self.members.items():
            if ip.n:
                ip.phi[...] = i_phi[i]

    def set_acc(self, other,
                kernel=ext.get_kernel('Acc')):
        """Set individual gravitational acceleration due to other particles.

        """
        i_acc = {i: 0 for i in self.members.keys()}
        j_acc = {j: 0 for j in other.members.keys()}

        for i, ip in self.members.items():
            if ip.n:
                for j, jp in other.members.items():
                    if jp.n:
                        kernel(ip, jp)
                        i_acc[i] += ip.rdot[2]
                        j_acc[j] += jp.rdot[2]

        if self != other:
            for j, jp in other.members.items():
                if jp.n:
                    jp.rdot[2] = j_acc[j]
        for i, ip in self.members.items():
            if ip.n:
                ip.rdot[2] = i_acc[i]

    def set_pnacc(self, other, pn=None, use_auxvel=False,
                  kernel=ext.get_kernel('PNAcc')):
        """Set individual post-Newtonian gravitational acceleration due to
        other particles.

        """
        def swap_vel(ps):
            for p in ps.members.values():
                v, w = p.rdot[1], p.pnvel
                vv, ww = v.copy(), w.copy()
                v[...], w[...] = ww, vv

        if use_auxvel:
            swap_vel(self)
            if self != other:
                swap_vel(other)

        kernel(self, other, pn=pn)

        if use_auxvel:
            swap_vel(self)
            if self != other:
                swap_vel(other)

    def set_acc_jrk(self, other,
                    kernel=ext.get_kernel('AccJrk')):
        """Set individual gravitational acceleration and jerk due to other
        particles.

        """
        i_acc = {i: 0 for i in self.members.keys()}
        j_acc = {j: 0 for j in other.members.keys()}
        i_jrk = {i: 0 for i in self.members.keys()}
        j_jrk = {j: 0 for j in other.members.keys()}

        for i, ip in self.members.items():
            if ip.n:
                for j, jp in other.members.items():
                    if jp.n:
                        kernel(ip, jp)
                        i_acc[i] += ip.rdot[2]
                        j_acc[j] += jp.rdot[2]
                        i_jrk[i] += ip.rdot[3]
                        j_jrk[j] += jp.rdot[3]

        if self != other:
            for j, jp in other.members.items():
                if jp.n:
                    jp.rdot[2] = j_acc[j]
                    jp.rdot[3] = j_jrk[j]
        for i, ip in self.members.items():
            if ip.n:
                ip.rdot[2] = i_acc[i]
                ip.rdot[3] = i_jrk[i]

    def set_snp_crk(self, other,
                    kernel=ext.get_kernel('SnpCrk')):
        """Set individual gravitational snap and crackle due to other
        particles.

        """
        i_acc = {i: 0 for i in self.members.keys()}
        j_acc = {j: 0 for j in other.members.keys()}
        i_jrk = {i: 0 for i in self.members.keys()}
        j_jrk = {j: 0 for j in other.members.keys()}
        i_snp = {i: 0 for i in self.members.keys()}
        j_snp = {j: 0 for j in other.members.keys()}
        i_crk = {i: 0 for i in self.members.keys()}
        j_crk = {j: 0 for j in other.members.keys()}

        for i, ip in self.members.items():
            if ip.n:
                for j, jp in other.members.items():
                    if jp.n:
                        kernel(ip, jp)
                        i_acc[i] += ip.rdot[2]
                        j_acc[j] += jp.rdot[2]
                        i_jrk[i] += ip.rdot[3]
                        j_jrk[j] += jp.rdot[3]
                        i_snp[i] += ip.rdot[4]
                        j_snp[j] += jp.rdot[4]
                        i_crk[i] += ip.rdot[5]
                        j_crk[j] += jp.rdot[5]

        if self != other:
            for j, jp in other.members.items():
                if jp.n:
                    jp.rdot[2] = j_acc[j]
                    jp.rdot[3] = j_jrk[j]
                    jp.rdot[4] = j_snp[j]
                    jp.rdot[5] = j_crk[j]
        for i, ip in self.members.items():
            if ip.n:
                ip.rdot[2] = i_acc[i]
                ip.rdot[3] = i_jrk[i]
                ip.rdot[4] = i_snp[i]
                ip.rdot[5] = i_crk[i]


# -- End of File --
