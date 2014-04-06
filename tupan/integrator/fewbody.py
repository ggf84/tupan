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
    def drift(ips, dt):
        """Drift operator for post-Newtonian quantities.

        """
        ips.rx += ips.vx * dt
        ips.ry += ips.vy * dt
        ips.rz += ips.vz * dt
        if ips.include_pn_corrections:
            ips.pn_drift_com_r(dt)
        return ips

    @staticmethod
    @timings
    def kepler_solver(ips, dt):
        """

        """
        if ips.include_pn_corrections:
            raise NotImplementedError("The current version of the "
                                      "Kepler-solver does not include "
                                      "post-Newtonian corrections.")
        else:
            extensions.kepler.calc(ips, ips, dt=dt)
        return ips

    @staticmethod
    @timings
    def evolve(ips, dt):
        """

        """
        if ips.n == 0:
            return ips

        if ips.n == 1:
            return FewBody.drift(ips, dt)

        # if ips.n == 2:
        return FewBody.kepler_solver(ips, dt)


# -- End of File --
