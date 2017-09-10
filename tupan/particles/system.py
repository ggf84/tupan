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

    def prepare_masks(self, func):
        masks = {}
        for name, ps in self.members.items():
            if ps.n:
                mask = func(ps)
                n0 = int(mask.sum())
                n1 = ps.n - n0
                masks[name] = (n0, n1, mask, ~mask)
        return masks

    def split_by(self, masks):
        data0, data1 = {}, {}
        for name, ps in self.members.items():
            if ps.n:
                n0, n1, mask0, mask1 = masks[name]
                if n0 and n1:
                    data0[name] = ps[mask0]
                    data1[name] = ps[mask1]
                elif not n1:
                    data0[name] = ps
                    data1[name] = ps.empty()
                elif not n0:
                    data0[name] = ps.empty()
                    data1[name] = ps
        return (self.from_members(**data0),
                self.from_members(**data1))

    def join_by(self, masks, ps0, ps1):
        for name, ps in self.members.items():
            if ps.n:
                n0, n1, mask0, mask1 = masks[name]
                if n0 and n1:
                    ps[mask0] = ps0.members[name]
                    ps[mask1] = ps1.members[name]

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
        if not other.n:
            return self
        if not self.n:
            return other
        members = {}
        for name in set([*self.members, *other.members]):
            if name not in other.members:
                members[name] = self.members[name]
            elif name not in self.members:
                members[name] = other.members[name]
            else:
                members[name] = self.members[name] + other.members[name]
        return self.from_members(**members)

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
        return float(
            min(abs(ps.time).min() for ps in self.members.values() if ps.n)
        )

    @property
    def tstep_min(self):
        return float(
            min(abs(ps.tstep).min() for ps in self.members.values() if ps.n)
        )

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
                if 'pn_mr' in ps.data:
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
                if 'pn_mv' in ps.data:
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
                if 'pn_mv' in ps.data:
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
                if 'pn_am' in ps.data:
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
                if 'pn_ke' in ps.data:
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
                nforce=1,
                kernel=ext.make_extension('Phi')):
        """Set individual gravitational potential due to other particles.

        """
        consts = kernel.set_consts()

        ibufs = {}
        for i, ip in self.members.items():
            if ip.n:
                ibufs[i] = kernel.set_bufs(ip, nforce=nforce)
        jbufs = {**ibufs}
        if self != other:
            for j, jp in other.members.items():
                if jp.n:
                    jbufs[j] = kernel.set_bufs(jp, nforce=nforce)

        interactions = []
        for i, ip in self.members.items():
            if ip.n:
                for j, jp in other.members.items():
                    if jp.n:
                        if ip == jp:
                            args = consts + ibufs[i]
                            kernel.triangle(*args)
                        elif (ip, jp) not in interactions:
                            args = consts + ibufs[i] + jbufs[j]
                            kernel.rectangle(*args)
                            interactions.append((jp, ip))

        for i, ip in self.members.items():
            if ip.n:
                kernel.map_bufs(ibufs[i], ip, nforce=nforce)
#                ip.phi[...] = ip.phi
        if self != other:
            for j, jp in other.members.items():
                if jp.n:
                    kernel.map_bufs(jbufs[j], jp, nforce=nforce)
#                    jp.phi[...] = jp.phi

    def set_acc(self, other,
                nforce=1,
                kernel=ext.make_extension('Acc')):
        """Set individual gravitational acceleration due to other particles.

        """
        consts = kernel.set_consts()

        ibufs = {}
        for i, ip in self.members.items():
            if ip.n:
                ibufs[i] = kernel.set_bufs(ip, nforce=nforce)
        jbufs = {**ibufs}
        if self != other:
            for j, jp in other.members.items():
                if jp.n:
                    jbufs[j] = kernel.set_bufs(jp, nforce=nforce)

        interactions = []
        for i, ip in self.members.items():
            if ip.n:
                for j, jp in other.members.items():
                    if jp.n:
                        if ip == jp:
                            args = consts + ibufs[i]
                            kernel.triangle(*args)
                        elif (ip, jp) not in interactions:
                            args = consts + ibufs[i] + jbufs[j]
                            kernel.rectangle(*args)
                            interactions.append((jp, ip))

        for i, ip in self.members.items():
            if ip.n:
                kernel.map_bufs(ibufs[i], ip, nforce=nforce)
                ip.rdot[2:2+nforce] = ip.fdot[:nforce]
        if self != other:
            for j, jp in other.members.items():
                if jp.n:
                    kernel.map_bufs(jbufs[j], jp, nforce=nforce)
                    jp.rdot[2:2+nforce] = jp.fdot[:nforce]

    def set_acc_jrk(self, other,
                    nforce=2,
                    kernel=ext.make_extension('Acc_Jrk')):
        """Set individual gravitational acceleration and jerk due to other
        particles.

        """
        consts = kernel.set_consts()

        ibufs = {}
        for i, ip in self.members.items():
            if ip.n:
                ibufs[i] = kernel.set_bufs(ip, nforce=nforce)
        jbufs = {**ibufs}
        if self != other:
            for j, jp in other.members.items():
                if jp.n:
                    jbufs[j] = kernel.set_bufs(jp, nforce=nforce)

        interactions = []
        for i, ip in self.members.items():
            if ip.n:
                for j, jp in other.members.items():
                    if jp.n:
                        if ip == jp:
                            args = consts + ibufs[i]
                            kernel.triangle(*args)
                        elif (ip, jp) not in interactions:
                            args = consts + ibufs[i] + jbufs[j]
                            kernel.rectangle(*args)
                            interactions.append((jp, ip))

        for i, ip in self.members.items():
            if ip.n:
                kernel.map_bufs(ibufs[i], ip, nforce=nforce)
                ip.rdot[2:2+nforce] = ip.fdot[:nforce]
        if self != other:
            for j, jp in other.members.items():
                if jp.n:
                    kernel.map_bufs(jbufs[j], jp, nforce=nforce)
                    jp.rdot[2:2+nforce] = jp.fdot[:nforce]

    def set_snp_crk(self, other,
                    nforce=4,
                    kernel=ext.make_extension('Snp_Crk')):
        """Set individual gravitational snap and crackle due to other
        particles.

        """
        consts = kernel.set_consts()

        ibufs = {}
        for i, ip in self.members.items():
            if ip.n:
                ibufs[i] = kernel.set_bufs(ip, nforce=nforce)
        jbufs = {**ibufs}
        if self != other:
            for j, jp in other.members.items():
                if jp.n:
                    jbufs[j] = kernel.set_bufs(jp, nforce=nforce)

        interactions = []
        for i, ip in self.members.items():
            if ip.n:
                for j, jp in other.members.items():
                    if jp.n:
                        if ip == jp:
                            args = consts + ibufs[i]
                            kernel.triangle(*args)
                        elif (ip, jp) not in interactions:
                            args = consts + ibufs[i] + jbufs[j]
                            kernel.rectangle(*args)
                            interactions.append((jp, ip))

        for i, ip in self.members.items():
            if ip.n:
                kernel.map_bufs(ibufs[i], ip, nforce=nforce)
                ip.rdot[2:2+nforce] = ip.fdot[:nforce]
        if self != other:
            for j, jp in other.members.items():
                if jp.n:
                    kernel.map_bufs(jbufs[j], jp, nforce=nforce)
                    jp.rdot[2:2+nforce] = jp.fdot[:nforce]

    def set_tstep(self, other, eta,
                  nforce=2,
                  kernel=ext.make_extension('Tstep')):
        """Set individual time-steps due to other particles.

        """
        consts = kernel.set_consts(eta=eta)

        ibufs = {}
        for i, ip in self.members.items():
            if ip.n:
                ibufs[i] = kernel.set_bufs(ip, nforce=nforce)
        jbufs = {**ibufs}
        if self != other:
            for j, jp in other.members.items():
                if jp.n:
                    jbufs[j] = kernel.set_bufs(jp, nforce=nforce)

        interactions = []
        for i, ip in self.members.items():
            if ip.n:
                for j, jp in other.members.items():
                    if jp.n:
                        if ip == jp:
                            args = consts + ibufs[i]
                            kernel.triangle(*args)
                        elif (ip, jp) not in interactions:
                            args = consts + ibufs[i] + jbufs[j]
                            kernel.rectangle(*args)
                            interactions.append((jp, ip))

        for i, ip in self.members.items():
            if ip.n:
                kernel.map_bufs(ibufs[i], ip, nforce=nforce)
                ip.tstep[...] = eta / np.sqrt(ip.tstep)
                ip.tstep_sum[...] = eta / np.sqrt(ip.tstep_sum)
        if self != other:
            for j, jp in other.members.items():
                if jp.n:
                    kernel.map_bufs(jbufs[j], jp, nforce=nforce)
                    jp.tstep[...] = eta / np.sqrt(jp.tstep)
                    jp.tstep_sum[...] = eta / np.sqrt(jp.tstep_sum)

    def set_pnacc(self, other, pn=None, use_auxvel=False,
                  nforce=2,
                  kernel=ext.make_extension('PNAcc')):
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

        consts = kernel.set_consts(order=pn['order'], clight=pn['clight'])

        ibufs = {}
        for i, ip in self.members.items():
            if ip.n:
                ibufs[i] = kernel.set_bufs(ip, nforce=nforce)
        jbufs = {**ibufs}
        if self != other:
            for j, jp in other.members.items():
                if jp.n:
                    jbufs[j] = kernel.set_bufs(jp, nforce=nforce)

        interactions = []
        for i, ip in self.members.items():
            if ip.n:
                for j, jp in other.members.items():
                    if jp.n:
                        if ip == jp:
                            args = consts + ibufs[i]
                            kernel.triangle(*args)
                        elif (ip, jp) not in interactions:
                            args = consts + ibufs[i] + jbufs[j]
                            kernel.rectangle(*args)
                            interactions.append((jp, ip))

        for i, ip in self.members.items():
            if ip.n:
                kernel.map_bufs(ibufs[i], ip, nforce=nforce)
#                ip.pnacc[...] = ip.pnacc
        if self != other:
            for j, jp in other.members.items():
                if jp.n:
                    kernel.map_bufs(jbufs[j], jp, nforce=nforce)
#                    jp.pnacc[...] = jp.pnacc

        if use_auxvel:
            swap_vel(self)
            if self != other:
                swap_vel(other)


# -- End of File --
