# -*- coding: utf-8 -*-
#

"""This module provides initial conditions for some few-body systems with
known numerical solutions.
"""


from ..particles.allparticles import ParticleSystem


def make_pythagorean():
    """
    Returns initial conditions for the pythagorean 3-body system.
    """
    ps = ParticleSystem(3)

    ps.mass = [3.0, 4.0, 5.0]

    ps.rx = [+1.0, -2.0, +1.0]
    ps.ry = [+3.0, -1.0, -1.0]
    ps.rz = [0.0, 0.0, 0.0]

    ps.vx = [0.0, 0.0, 0.0]
    ps.vy = [0.0, 0.0, 0.0]
    ps.vz = [0.0, 0.0, 0.0]

    ps.id = range(ps.n)
    return ps


def make_circular3():
    """
    Returns initial conditions for a 3-body system in a circular orbit.
    """
    ps = ParticleSystem(3)

    ps.mass = [1.0, 1.0, 1.0]

    ps.rx = [0.0, +1.0, +0.5]
    ps.ry = [0.0, 0.0, +0.8660254037844386]
    ps.rz = [0.0, 0.0, 0.0]

    ps.vx = [-0.5, -0.5, +1.0]
    ps.vy = [+0.8660254037844386, -0.8660254037844386, 0.0]
    ps.vz = [0.0, 0.0, 0.0]

    ps.id = range(ps.n)
    return ps


def make_figure83():
    """
    Returns initial conditions for a 3-body system in a 8-shaped orbit.
    """
    ps = ParticleSystem(3)

    ps.mass = [1.0, 1.0, 1.0]

    ps.rx = [+0.9700436, -0.9700436, 0.0]
    ps.ry = [-0.24308753, +0.24308753, 0.0]
    ps.rz = [0.0, 0.0, 0.0]

    ps.vx = [+0.466203685, +0.466203685, -0.93240737]
    ps.vy = [+0.43236573, +0.43236573, -0.86473146]
    ps.vz = [0.0, 0.0, 0.0]

    ps.id = range(ps.n)
    return ps


def make_figure84():
    """
    Returns initial conditions for a 4-body system in a 8-shaped orbit.
    """
    ps = ParticleSystem(4)

    ps.mass = [1.0, 1.0, 1.0, 1.0]

    ps.rx = [+1.382857, 0.0, -1.382857, 0.0]
    ps.ry = [0.0, +0.157030, 0.0, -0.157030]
    ps.rz = [0.0, 0.0, 0.0, 0.0]

    ps.vx = [0.0, +1.871935, 0.0, -1.871935]
    ps.vy = [+0.584873, 0.0, -0.584873, 0.0]
    ps.vz = [0.0, 0.0, 0.0, 0.0]

    ps.id = range(ps.n)
    return ps


########## end of file ##########
