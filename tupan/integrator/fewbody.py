# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from ..lib import extensions
from ..lib.utils.timing import timings, bind_all


@bind_all(timings)
class FewBody(object):
    """

    """
    @staticmethod
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
    def kepler_solver(ips, dt):
        """

        """
        if ips.include_pn_corrections:
            raise NotImplementedError("The current version of the "
                                      "Kepler-solver does not include "
                                      "post-Newtonian corrections.")
        else:
            extensions.kepler(ips, ips, dt=dt)
        return ips

    @classmethod
    def evolve(cls, ips, dt):
        """

        """
        if ips.n == 0:
            return ips

        if ips.n == 1:
            return cls.drift(ips, dt)

        # if ips.n == 2:
        return cls.kepler_solver(ips, dt)


# -- End of File --
