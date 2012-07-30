#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import numpy as np
from .pbase import Pbase, make_attrs
from ..lib import gravity
from ..lib.utils.timing import decallmethods, timings


__all__ = ["BlackHole"]


@decallmethods(timings)
@make_attrs
class BlackHole(Pbase):
    """
    A base class for BlackHoles.
    """
    specific_attributes = [# name, dtype, doc
                           ('radius', 'f8', 'radius'),
                           ('pnacc', '3f8', 'post-Newtonian acceleration'),
                           ('spin', '3f8', 'spin'),
                           # auxiliary attributes
                           ('pncorrection_energy', 'f8', 'post-Newtonian correction for the energy'),
                           ('pncorrection_linear_momentum', '3f8', 'post-Newtonian correction for the linear momentum'),
                           ('pncorrection_angular_momentum', '3f8', 'post-Newtonian correction for the angular momentum'),
                           ('pncorrection_center_of_mass_position', '3f8', 'post-Newtonian correction for the center-of-mass position'),
                          ]

    specific_names = [_[0] for _ in specific_attributes]

    attributes = Pbase.common_attributes + specific_attributes
    names = Pbase.common_names + specific_names

    dtype = [(_[0], _[1]) for _ in attributes]
    data0 = np.zeros(0, dtype)


    #
    # specific methods
    #

    ### pn-gravity

    def update_pnacc(self, objs, pn_order, clight):
        """
        Update individual post-newtonian gravitational acceleration due j-particles.
        """
        gravity.pnacc.set_args(self, objs, pn_order, clight)
        gravity.pnacc.run()
        self.pnacc = gravity.pnacc.get_result()


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
