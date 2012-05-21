#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import numpy as np
from .pbase import Pbase
from ..lib.gravity import gravitation
from ..lib.utils.timing import decallmethods, timings


__all__ = ["BlackHole"]


@decallmethods(timings)
class BlackHole(Pbase):
    """
    A base class for BlackHoles.
    """
    attrs = ["id", "mass", "pos", "vel", "acc", "spin", "phi",
             "eps2", "t_curr", "dt_prev", "dt_next"]
    dtype = [# common attributes
             ("id", "u8"),
             ("mass", "f8"),
             ("pos", "3f8"),
             ("vel", "3f8"),
             ("acc", "3f8"),
             ("phi", "f8"),
             ("eps2", "f8"),
             ("t_curr", "f8"),
             ("dt_prev", "f8"),
             ("dt_next", "f8"),
             # specific attributes
             ("radius", "f8"),
             ("pnacc", "3f8"),
             ("spin", "3f8"),
             # auxiliary attributes
             ("pncorrection_energy", "f8"),
             ("pncorrection_linear_momentum", "3f8"),
             ("pncorrection_angular_momentum", "3f8"),
             ("pncorrection_center_of_mass_position", "3f8"),
            ]

    zero = np.zeros(0, dtype)

    def __init__(self, n=0):
        super(BlackHole, self).__init__(n)


    #
    # specific attributes
    #

    ### radius

    @property
    def radius(self):
        return self.data['radius']

    @radius.setter
    def radius(self, values):
        self.data['radius'] = values

    @radius.deleter
    def radius(self):
        raise NotImplementedError()


    ### pnacc

    @property
    def pnacc(self):
        return self.data['pnacc']

    @pnacc.setter
    def pnacc(self, values):
        self.data['pnacc'] = values

    @pnacc.deleter
    def pnacc(self):
        raise NotImplementedError()


    ### spin

    @property
    def spin(self):
        return self.data['spin']

    @spin.setter
    def spin(self, values):
        self.data['spin'] = values

    @spin.deleter
    def spin(self):
        raise NotImplementedError()


    #
    # auxiliary attributes
    #

    ### pncorrection_energy

    @property
    def pncorrection_energy(self):
        return self.data['pncorrection_energy']

    @pncorrection_energy.setter
    def pncorrection_energy(self, values):
        self.data['pncorrection_energy'] = values

    @pncorrection_energy.deleter
    def pncorrection_energy(self):
        raise NotImplementedError()


    ### pncorrection_linear_momentum

    @property
    def pncorrection_linear_momentum(self):
        return self.data['pncorrection_linear_momentum']

    @pncorrection_linear_momentum.setter
    def pncorrection_linear_momentum(self, values):
        self.data['pncorrection_linear_momentum'] = values

    @pncorrection_linear_momentum.deleter
    def pncorrection_linear_momentum(self):
        raise NotImplementedError()


    ### pncorrection_angular_momentum

    @property
    def pncorrection_angular_momentum(self):
        return self.data['pncorrection_angular_momentum']

    @pncorrection_angular_momentum.setter
    def pncorrection_angular_momentum(self, values):
        self.data['pncorrection_angular_momentum'] = values

    @pncorrection_angular_momentum.deleter
    def pncorrection_angular_momentum(self):
        raise NotImplementedError()


    ### pncorrection_center_of_mass_position

    @property
    def pncorrection_center_of_mass_position(self):
        return self.data['pncorrection_center_of_mass_position']

    @pncorrection_center_of_mass_position.setter
    def pncorrection_center_of_mass_position(self, values):
        self.data['pncorrection_center_of_mass_position'] = values

    @pncorrection_center_of_mass_position.deleter
    def pncorrection_center_of_mass_position(self):
        raise NotImplementedError()


    #
    # specific methods
    #

    ### pn-gravity

    def update_pnacc(self, jobj, pn_order, clight):
        self.pnacc = self.get_pnacc(jobj, pn_order, clight)

    def get_pnacc(self, jobj, pn_order, clight):
        """
        Get individual post-newtonian gravitational acceleration due j-particles.
        """
        if jobj.n:
            return gravitation.set_pnacc(self, jobj, pn_order, clight)
        return 0.0


    ### evolve corrections due to post-newtonian terms

    def evolve_velocity_correction_due_to_pnterms(self, tstep):
        """
        Evolves velocity correction in time due to post-newtonian terms.
        """
        self.vel += tstep * self.pnacc

    def evolve_energy_correction_due_to_pnterms(self, tstep):
        """
        Evolves energy correction in time due to post-newtonian terms.
        """
        pnforce = -(self.mass * self.pnacc.T).T
        e_jump = tstep * (self.vel * pnforce).sum(1)
        self.pncorrection_energy += e_jump

    def evolve_linear_momentum_correction_due_to_pnterms(self, tstep):
        """
        Evolves linear momentum correction in time due to post-newtonian terms.
        """
        pnforce = -(self.mass * self.pnacc.T).T
        lm_jump = tstep * pnforce
        self.pncorrection_linear_momentum += lm_jump

    def evolve_angular_momentum_correction_due_to_pnterms(self, tstep):
        """
        Evolves angular momentum correction in time due to post-newtonian terms.
        """
        pnforce = -(self.mass * self.pnacc.T).T
        am_jump = tstep * np.cross(self.pos, pnforce)
        self.pncorrection_angular_momentum += am_jump

    def evolve_center_of_mass_position_correction_due_to_pnterms(self, tstep):
        """
        Evolves center of mass position correction in time due to post-newtonian terms.
        """
        comp_jump = tstep * self.pncorrection_linear_momentum
        self.pncorrection_center_of_mass_position += comp_jump


    #
    # auxiliary methods
    #

    def get_pn_correction_for_total_energy(self):
        return self.pncorrection_energy.sum(0)

    def get_pn_correction_for_total_linear_momentum(self):
        return self.pncorrection_linear_momentum.sum(0)

    def get_pn_correction_for_total_angular_momentum(self):
        return self.pncorrection_angular_momentum.sum(0)

    def get_pn_correction_for_center_of_mass_position(self):
        return self.pncorrection_center_of_mass_position.sum(0)

    def get_pn_correction_for_center_of_mass_velocity(self):
        return self.get_pn_correction_for_total_linear_momentum()


    #
    # overridden methods
    #

    ### ...


########## end of file ##########
