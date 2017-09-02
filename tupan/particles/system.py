# -*- coding: utf-8 -*-
#

"""
TODO.
"""

import copy
import numpy as np
from .base import (Bodies, Stars, Planets, Blackholes, Gas)
from ..lib import extensions as ext


class ParticleSystem(object):
    """
    This class holds the particle system in the simulation.
    """
    def __init__(self, nbodies=0, nstars=0, nplanets=0, nblackholes=0, ngas=0):
        members = {cls.name: cls(n)
                   for (n, cls) in [(nbodies, Bodies),
                                    (nstars, Stars),
                                    (nplanets, Planets),
                                    (nblackholes, Blackholes),
                                    (ngas, Gas)] if n}
        self.update_members(**members)
        if self.n:
            self.reset_pid()

    def update_members(self, **members):
        dct = vars(self)
        dct.clear()
        dct.update(**members)
        self.members = members
        self.n = len(self)

    @classmethod
    def from_members(cls, **members):
        obj = cls.__new__(cls)
        obj.update_members(**members)
        return obj

    def reset_pid(self):
        for ps in self.members.values():
            if ps.n:
                ps.pid[...] = range(ps.n)

    def astype(self, name='bodies'):
        cls = globals()[name.capitalize()]
        obj = cls()
        for ps in self.members.values():
            if ps.n:
                obj += ps.astype(cls)
        return self.from_members(**{cls.name: obj})

    def split_by(self, mask_function):
        a, b = {}, {}
        for name, ps in self.members.items():
            if ps.n:
                a[name], b[name] = ps.split_by(mask_function)
        return (self.from_members(**a),
                self.from_members(**b))

    def copy(self):
        return copy.deepcopy(self)

    #
    # miscellaneous methods
    #

    def __len__(self):
        return sum(len(ps) for ps in self.members.values())

    def __repr__(self):
        return repr(vars(self))

    def __str__(self):
        fmt = self.__class__.__name__ + '(['
        for ps in self.members.values():
            if ps.n:
                fmt += '\n\t{0},'.format('\n\t'.join(str(ps).split('\n')))
        if self.n:
            fmt += '\n'
        fmt += '])'
        return fmt

    def __add__(self, other):
        if other.n:
            members = self.members
            for name, ps in other.members.items():
                try:
                    members[name] += ps
                except KeyError:
                    members[name] = ps.copy()
            return self.from_members(**members)
        return self

    __radd__ = __add__

    def __getitem__(self, index):
        if isinstance(index, np.ndarray):
            ns, nf = 0, 0
            members = {}
            for name, ps in self.members.items():
                nf += ps.n
                members[name] = ps[index[ns:nf]]
                ns += ps.n
            return self.from_members(**members)

        if isinstance(index, int):
            if abs(index) > self.n-1:
                msg = 'index {0} out of bounds 0<=index<{1}'
                raise IndexError(msg.format(index, self.n))
            if index < 0:
                index = self.n + index
            members = {}
            for name, ps in self.members.items():
                if 0 <= index < ps.n:
                    members[name] = ps[index]
                index -= ps.n
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
            for name, ps in self.members.items():
                if stop >= 0 and start < ps.n:
                    members[name] = ps[start-ps.n:stop]
                start -= ps.n
                stop -= ps.n
            return self.from_members(**members)

    def __setitem__(self, index, value):
        if isinstance(index, np.ndarray):
            ns, nf = 0, 0
            for name, ps in self.members.items():
                nf += ps.n
                if name in value.members:
                    ps[index[ns:nf]] = value.members[name]
                ns += ps.n
            return

        if isinstance(index, int):
            if abs(index) > self.n-1:
                msg = 'index {0} out of bounds 0<=index<{1}'
                raise IndexError(msg.format(index, self.n))
            if index < 0:
                index = self.n + index
            for name, ps in self.members.items():
                if 0 <= index < ps.n:
                    if name in value.members:
                        ps[index] = value.members[name]
                index -= ps.n
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
            for name, ps in self.members.items():
                if stop >= 0 and start < ps.n:
                    if name in value.members:
                        ps[start-ps.n:stop] = value.members[name]
                start -= ps.n
                stop -= ps.n
            return

    #
    # n-body methods
    #

    @property
    def global_time(self):
        return min(abs(ps.time).min() for ps in self.members.values() if ps.n)

    @property
    def tstep_min(self):
        return min(abs(ps.tstep).min() for ps in self.members.values() if ps.n)

    # -- total mass and center-of-mass
    @property
    def total_mass(self):
        """Total mass of the system.

        """
        return float(sum(ps.mass.sum() for ps in self.members.values()))

    @property
    def com_r(self):
        """Center-of-Mass position of the system.

        .. note::

            Post-Newtonian corrections, if enabled, are included.

        """
        mr_sum = 0
        for ps in self.members.values():
            if ps.n:
                mr = ps.mass * ps.rdot[0]
                if hasattr(ps, 'pn_mr'):
                    mr += ps.pn_mr
                mr_sum += mr.sum(1)
        return mr_sum / self.total_mass

    @property
    def com_v(self):
        """Center-of-Mass velocity of the system.

        .. note::

            Post-Newtonian corrections, if enabled, are included.

        """
        mv_sum = 0
        for ps in self.members.values():
            if ps.n:
                mv = ps.mass * ps.rdot[1]
                if hasattr(ps, 'pn_mv'):
                    mv += ps.pn_mv
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
        for ps in self.members.values():
            if ps.n:
                ps.rdot[0].T[...] += com_r
                ps.rdot[1].T[...] += com_v

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
        for ps in self.members.values():
            if ps.n:
                lm = ps.mass * ps.rdot[1]
                if hasattr(ps, 'pn_mv'):
                    lm += ps.pn_mv
                lm_sum += lm.sum(1)
        return lm_sum

    @property
    def angular_momentum(self):
        """Total angular momentum of the system.

        .. note::

            Post-Newtonian corrections, if enabled, are included.

        """
        am_sum = 0
        for ps in self.members.values():
            if ps.n:
                mv = ps.mass * ps.rdot[1]
                am = np.cross(ps.rdot[0].T, mv.T).T
                if hasattr(ps, 'pn_am'):
                    am += ps.pn_am
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
        for ps in self.members.values():
            if ps.n:
                v2 = (ps.rdot[1]**2).sum(0)
                ke = 0.5 * ps.mass * v2
                if hasattr(ps, 'pn_ke'):
                    ke += ps.pn_ke
                ke_sum += ke.sum()
        return float(ke_sum)

    @property
    def potential_energy(self):
        """Total potential energy.

        """
        pe_sum = 0
        for ps in self.members.values():
            if ps.n:
                pe = ps.mass * ps.phi
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
        for ps in self.members.values():
            if ps.n:
                pos = (ps.rdot[0].T - com_r).T
                mr2 = ps.mass * pos**2
                mr2_sum += mr2.sum()
        return (mr2_sum / self.total_mass)**0.5

    # -- rescaling methods
    def dynrescale_total_mass(self, total_mass):
        """Rescales the total mass of the system while maintaining its
        dynamics unchanged.

        """
        m_ratio = total_mass / self.total_mass
        for ps in self.members.values():
            if ps.n:
                ps.mass *= m_ratio
                ps.rdot[0] *= m_ratio

    def dynrescale_radial_size(self, size):
        """Rescales the radial size of the system while maintaining its
        dynamics unchanged.

        """
        r_scale = size / self.radial_size
        v_scale = 1 / r_scale**0.5
        for ps in self.members.values():
            if ps.n:
                ps.rdot[0] *= r_scale
                ps.rdot[1] *= v_scale

    def dynrescale_virial_radius(self, rvir):
        """Rescales the virial radius of the system while maintaining its
        dynamics unchanged.

        """
        self.set_phi(self)
        r_scale = rvir / self.virial_radius
        v_scale = 1 / r_scale**0.5
        for ps in self.members.values():
            if ps.n:
                ps.rdot[0] *= r_scale
                ps.rdot[1] *= v_scale

    def scale_to_virial(self):
        """Rescale system to virial equilibrium (2K + U = 0).

        """
        self.set_phi(self)
        ke = self.kinetic_energy
        pe = self.potential_energy
        v_scale = ((-0.5 * pe) / ke)**0.5
        for ps in self.members.values():
            if ps.n:
                ps.rdot[1] *= v_scale

    def to_nbody_units(self):
        """Rescales system to nbody units while maintaining its dynamics
        unchanged.

        """
        self.dynrescale_total_mass(1.0)
        self.dynrescale_virial_radius(1.0)

    # -- O(N^2) methods
    def set_phi(self, other,
                kernel_r=ext.make_extension('Phi_rectangle'),
                kernel_t=ext.make_extension('Phi_triangle')):
        """Set individual gravitational potential due to other particles.

        """
        iphi = {i: 0 for i in self.members.keys()}
        jphi = {j: 0 for j in other.members.keys()}

        interactions = []
        for i, ip in self.members.items():
            if ip.n:
                for j, jp in other.members.items():
                    if jp.n:
                        if ip == jp:
                            kernel_t(ip)
                            iphi[i] += ip.phi
                        elif (ip, jp) not in interactions:
                            interactions.append((jp, ip))
                            kernel_r(ip, jp)
                            iphi[i] += ip.phi
                            jphi[j] += jp.phi
                            if self == other:
                                jphi[i] += ip.phi
                                iphi[j] += jp.phi

        if self != other:
            for j, jp in other.members.items():
                if jp.n:
                    jp.phi[...] = jphi[j]
        for i, ip in self.members.items():
            if ip.n:
                ip.phi[...] = iphi[i]

    def set_acc(self, other,
                kernel_r=ext.make_extension('Acc_rectangle'),
                kernel_t=ext.make_extension('Acc_triangle')):
        """Set individual gravitational acceleration due to other particles.

        """
        nforce = 1
        ifdot = {i: 0 for i in self.members.keys()}
        jfdot = {j: 0 for j in other.members.keys()}

        interactions = []
        for i, ip in self.members.items():
            if ip.n:
                for j, jp in other.members.items():
                    if jp.n:
                        if ip == jp:
                            kernel_t(ip, nforce=nforce)
                            ifdot[i] += ip.fdot[:nforce]
                        elif (ip, jp) not in interactions:
                            interactions.append((jp, ip))
                            kernel_r(ip, jp, nforce=nforce)
                            ifdot[i] += ip.fdot[:nforce]
                            jfdot[j] += jp.fdot[:nforce]
                            if self == other:
                                jfdot[i] += ip.fdot[:nforce]
                                ifdot[j] += jp.fdot[:nforce]

        if self != other:
            for j, jp in other.members.items():
                if jp.n:
                    jp.rdot[2:2+nforce] = jfdot[j]
        for i, ip in self.members.items():
            if ip.n:
                ip.rdot[2:2+nforce] = ifdot[i]

    def set_acc_jrk(self, other,
                    kernel_r=ext.make_extension('AccJrk_rectangle'),
                    kernel_t=ext.make_extension('AccJrk_triangle')):
        """Set individual gravitational acceleration and jerk due to other
        particles.

        """
        nforce = 2
        ifdot = {i: 0 for i in self.members.keys()}
        jfdot = {j: 0 for j in other.members.keys()}

        interactions = []
        for i, ip in self.members.items():
            if ip.n:
                for j, jp in other.members.items():
                    if jp.n:
                        if ip == jp:
                            kernel_t(ip, nforce=nforce)
                            ifdot[i] += ip.fdot[:nforce]
                        elif (ip, jp) not in interactions:
                            interactions.append((jp, ip))
                            kernel_r(ip, jp, nforce=nforce)
                            ifdot[i] += ip.fdot[:nforce]
                            jfdot[j] += jp.fdot[:nforce]
                            if self == other:
                                jfdot[i] += ip.fdot[:nforce]
                                ifdot[j] += jp.fdot[:nforce]

        if self != other:
            for j, jp in other.members.items():
                if jp.n:
                    jp.rdot[2:2+nforce] = jfdot[j]
        for i, ip in self.members.items():
            if ip.n:
                ip.rdot[2:2+nforce] = ifdot[i]

    def set_snp_crk(self, other,
                    kernel_r=ext.make_extension('SnpCrk_rectangle'),
                    kernel_t=ext.make_extension('SnpCrk_triangle')):
        """Set individual gravitational snap and crackle due to other
        particles.

        """
        nforce = 4
        ifdot = {i: 0 for i in self.members.keys()}
        jfdot = {j: 0 for j in other.members.keys()}

        interactions = []
        for i, ip in self.members.items():
            if ip.n:
                for j, jp in other.members.items():
                    if jp.n:
                        if ip == jp:
                            kernel_t(ip, nforce=nforce)
                            ifdot[i] += ip.fdot[:nforce]
                        elif (ip, jp) not in interactions:
                            interactions.append((jp, ip))
                            kernel_r(ip, jp, nforce=nforce)
                            ifdot[i] += ip.fdot[:nforce]
                            jfdot[j] += jp.fdot[:nforce]
                            if self == other:
                                jfdot[i] += ip.fdot[:nforce]
                                ifdot[j] += jp.fdot[:nforce]

        if self != other:
            for j, jp in other.members.items():
                if jp.n:
                    jp.rdot[2:2+nforce] = jfdot[j]
        for i, ip in self.members.items():
            if ip.n:
                ip.rdot[2:2+nforce] = ifdot[i]

    def set_tstep(self, other, eta,
                  kernel_r=ext.make_extension('Tstep_rectangle'),
                  kernel_t=ext.make_extension('Tstep_triangle')):
        """Set individual time-steps due to other particles.

        """
        itstep = {i: 0 for i in self.members.keys()}
        jtstep = {j: 0 for j in other.members.keys()}
        itstep_sum = {i: 0 for i in self.members.keys()}
        jtstep_sum = {j: 0 for j in other.members.keys()}

        interactions = []
        for i, ip in self.members.items():
            if ip.n:
                for j, jp in other.members.items():
                    if jp.n:
                        if ip == jp:
                            kernel_t(ip, eta=eta)
                            itstep_sum[i] += ip.tstep_sum
                            itstep[i] = np.maximum(itstep[i], ip.tstep)
                        elif (ip, jp) not in interactions:
                            interactions.append((jp, ip))
                            kernel_r(ip, jp, eta=eta)
                            itstep_sum[i] += ip.tstep_sum
                            itstep[i] = np.maximum(itstep[i], ip.tstep)
                            jtstep_sum[j] += jp.tstep_sum
                            jtstep[j] = np.maximum(jtstep[j], jp.tstep)
                            if self == other:
                                jtstep_sum[i] += ip.tstep_sum
                                jtstep[i] = np.maximum(jtstep[i], ip.tstep)
                                itstep_sum[j] += jp.tstep_sum
                                itstep[j] = np.maximum(itstep[j], jp.tstep)

        if self != other:
            for j, jp in other.members.items():
                if jp.n:
                    jp.tstep[...] = eta / np.sqrt(jtstep[j])
                    jp.tstep_sum[...] = eta / np.sqrt(jtstep_sum[j])
        for i, ip in self.members.items():
            if ip.n:
                ip.tstep[...] = eta / np.sqrt(itstep[i])
                ip.tstep_sum[...] = eta / np.sqrt(itstep_sum[i])

    def set_pnacc(self, other, pn=None, use_auxvel=False,
                  kernel_r=ext.make_extension('PNAcc_rectangle'),
                  kernel_t=ext.make_extension('PNAcc_triangle')):
        """Set individual post-Newtonian gravitational acceleration due to
        other particles.

        """
        def swap_vel(ps):
            for ps in ps.members.values():
                v, w = ps.rdot[1], ps.pnvel
                vv, ww = v.copy(), w.copy()
                v[...], w[...] = ww, vv

        if use_auxvel:
            swap_vel(self)
            if self != other:
                swap_vel(other)

        ipnacc = {i: 0 for i in self.members.keys()}
        jpnacc = {j: 0 for j in other.members.keys()}

        interactions = []
        for i, ip in self.members.items():
            if ip.n:
                for j, jp in other.members.items():
                    if jp.n:
                        if ip == jp:
                            kernel_t(ip, pn=pn)
                            ipnacc[i] += ip.pnacc
                        elif (ip, jp) not in interactions:
                            interactions.append((jp, ip))
                            kernel_r(ip, jp, pn=pn)
                            ipnacc[i] += ip.pnacc
                            jpnacc[j] += jp.pnacc
                            if self == other:
                                jpnacc[i] += ip.pnacc
                                ipnacc[j] += jp.pnacc

        if self != other:
            for j, jp in other.members.items():
                if jp.n:
                    jp.pnacc[...] = jpnacc[j]
        for i, ip in self.members.items():
            if ip.n:
                ip.pnacc[...] = ipnacc[i]

        if use_auxvel:
            swap_vel(self)
            if self != other:
                swap_vel(other)


# -- End of File --
