# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from ..lib import extensions
from ..lib.utils.timing import timings


class FewBody(object):
    """

    """
    @staticmethod
    @timings
    def drift(ips, tau):
        """Drift operator for post-Newtonian quantities.

        """
        ips.rx += ips.vx * tau
        ips.ry += ips.vy * tau
        ips.rz += ips.vz * tau
        if ips.include_pn_corrections:
            ips.pn_drift_com_r(tau)
        return ips

    @staticmethod
    @timings
    def kepler_solver(ips, tau):
        """

        """
        if ips.include_pn_corrections:
            raise NotImplementedError("The current version of the "
                                      "Kepler-solver does not include "
                                      "post-Newtonian corrections.")
        else:
            extensions.kepler.calc(ips, ips, tau)
        return ips

    @staticmethod
    @timings
    def evolve(ips, tau):
        """

        """
        if ips.n == 0:
            return ips

        if ips.n == 1:
            return FewBody.drift(ips, tau)

        # if ips.n == 2:
        return FewBody.kepler_solver(ips, tau)


# -- End of File --
