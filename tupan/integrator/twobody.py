# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from ..lib import extensions


def kepler_solver(ps, tau):
    if ps.include_pn_corrections:
        raise NotImplementedError("The current version of the Kepler-solver does"
                                  " not include post-Newtonian corrections.")
    else:
        extensions.kepler.calc(ps, ps, tau)
    return ps


########## end of file ##########
