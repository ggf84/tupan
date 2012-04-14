#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import numpy as np
from .pbase import Pbase


__all__ = ["BlackHole"]


class BlackHole(Pbase):
    """
    A base class for BlackHoles.
    """
    dtype = [# common attributes
             ("key", "u8"),
             ("mass", "f8"),
             ("pos", "3f8"),
             ("vel", "3f8"),
             ("acc", "3f8"),
             ("phi", "f8"),
             ("eps2", "f8"),
             ("tcurr", "f8"),
             ("tnext", "f8"),
             # specific attributes
             ("pnacc", "3f8"),
             ("spin", "3f8"),
             ("radius", "f8"),
             # auxiliary attributes
             ("tstep", "f8"),
             ("pncorrection_energy", "f8"),
             ("pncorrection_linear_momentum", "3f8"),
             ("pncorrection_angular_momentum", "3f8"),
            ]

    def __init__(self, n=0):
        self.pncorrection_center_of_mass_position = np.zeros(3, "f8")
        self.pncorrection_center_of_mass_velocity = np.zeros(3, "f8")
        super(BlackHole, self).__init__(self.dtype, n)


    #
    # specific attributes
    #

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


    #
    # auxiliary attributes
    #

    ### tstep

    @property
    def tstep(self):
        return self.data['tstep']

    @tstep.setter
    def tstep(self, values):
        self.data['tstep'] = values

    @tstep.deleter
    def tstep(self):
        raise NotImplementedError()


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


    #
    # specific methods
    #

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

    def evolve_center_of_mass_position_correction_due_to_pnterms(self, tstep):
        """
        Evolves center of mass position correction in time due to post-newtonian terms.
        """
        comp_jump = tstep * self.pncorrection_linear_momentum.sum(0)
        self.pncorrection_center_of_mass_position += comp_jump

    def evolve_center_of_mass_velocity_correction_due_to_pnterms(self, tstep):
        """
        Evolves center of mass velocity correction in time due to post-newtonian terms.
        """
        pnforce = -(self.mass * self.pnacc.T).T
        comv_jump = tstep * pnforce.sum(0)
        self.pncorrection_center_of_mass_velocity += comv_jump

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


    #
    # auxiliary methods
    #

    def get_pn_correction_for_total_energy(self):
        return self.pncorrection_energy.sum(0)

    def get_pn_correction_for_center_of_mass_position(self):
        return self.pncorrection_center_of_mass_position

    def get_pn_correction_for_center_of_mass_velocity(self):
        return self.pncorrection_center_of_mass_velocity

    def get_pn_correction_for_total_linear_momentum(self):
        return self.pncorrection_linear_momentum.sum(0)

    def get_pn_correction_for_total_angular_momentum(self):
        return self.pncorrection_angular_momentum.sum(0)


    #
    # overridden methods
    #

    ### ...


########## end of file ##########
