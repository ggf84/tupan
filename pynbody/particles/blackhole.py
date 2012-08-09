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
    specific_attrs = [# name, dtype, doc
                      ('radius', 'f8', 'radius'),
                      ('pnacc', '3f8', 'post-Newtonian acceleration'),
                      ('spin', '3f8', 'spin'),
                      # auxiliary attributes
                      ('ke_pn_shift', 'f8', 'post-Newtonian correction for the kinetic energy'),
                      ('lmom_pn_shift', '3f8', 'post-Newtonian correction for the linear momentum'),
                      ('amom_pn_shift', '3f8', 'post-Newtonian correction for the angular momentum'),
                      ('rcom_pn_shift', '3f8', 'post-Newtonian correction for the center-of-mass position'),
                     ]

    specific_names = [_[0] for _ in specific_attrs]

    attrs = Pbase.common_attrs + specific_attrs
    names = Pbase.common_names + specific_names

    dtype = [(_[0], _[1]) for _ in attrs]
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


    ### evolves shift in quantities due to post-newtonian terms

    def evolve_ke_pn_shift(self, tstep):
        """
        Evolves kinetic energy shift in time due to post-newtonian terms.
        """
        pnforce = -(self.mass * self.pnacc.T).T
        ke_jump = tstep * (self.vel * pnforce).sum(1)
        self.ke_pn_shift += ke_jump

    def evolve_lmom_pn_shift(self, tstep):
        """
        Evolves linear momentum shift in time due to post-newtonian terms.
        """
        pnforce = -(self.mass * self.pnacc.T).T
        lmom_jump = tstep * pnforce
        self.lmom_pn_shift += lmom_jump

    def evolve_amom_pn_shift(self, tstep):
        """
        Evolves angular momentum shift in time due to post-newtonian terms.
        """
        pnforce = -(self.mass * self.pnacc.T).T
        amom_jump = tstep * np.cross(self.pos, pnforce)
        self.amom_pn_shift += amom_jump

    def evolve_rcom_pn_shift(self, tstep):
        """
        Evolves center of mass position shift in time due to post-newtonian terms.
        """
        rcom_jump = tstep * self.lmom_pn_shift
        self.rcom_pn_shift += rcom_jump


    #
    # auxiliary methods
    #

    def get_ke_pn_shift(self):
        return self.ke_pn_shift.sum(0)

    def get_lmom_pn_shift(self):
        return self.lmom_pn_shift.sum(0)

    def get_amom_pn_shift(self):
        return self.amom_pn_shift.sum(0)

    def get_rcom_pn_shift(self):
        return self.rcom_pn_shift.sum(0)

    def get_vcom_pn_shift(self):
        return self.lmom_pn_shift.sum(0)


    #
    # overridden methods
    #

    ### ...


########## end of file ##########
