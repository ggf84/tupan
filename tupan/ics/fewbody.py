# -*- coding: utf-8 -*-
#

"""This module provides initial conditions for some few-body systems with
known numerical solutions.
"""


from __future__ import division
from ..particles.allparticles import ParticleSystem


def make_binary(m1, m2, a, e):
    """
    Returns initial conditions for a binary system.
    """
    if not m1 > 0:
        raise ValueError("mass out of range (m1 > 0)")

    if not m2 > 0:
        raise ValueError("mass out of range (m2 > 0)")

    if not a > 0:
        raise ValueError("semi-major-axis out of range (a > 0)")

    if e > 1 or e < 0:
        raise ValueError("eccentricity out of range (0 < e < 1)")

    m = m1 + m2
    r = a * (1 + e)
    v = ((m / a) * (1 - e) / (1 + e))**0.5
    r1 = (m2 / m) * r
    r2 = -(m1 / m) * r
    v1 = (m2 / m) * v
    v2 = -(m1 / m) * v

    ps = ParticleSystem(2)

    ps.mass = [m1, m2]

    ps.rx = [r1, r2]
    ps.ry = [0.0, 0.0]
    ps.rz = [0.0, 0.0]

    ps.vx = [0.0, 0.0]
    ps.vy = [v1, v2]
    ps.vz = [0.0, 0.0]

    ps.id = range(ps.n)
    return ps


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


def make_solar_system():
    """
    Returns initial conditions for the 10-body solar system.
    """
    ps = ParticleSystem(10)

    ps.mass = [1.0,                     # Sun
               1.66013679527193e-07,    # Mercury
               2.44783833966455e-06,    # Venus
               3.04043264626853e-06,    # Earth
               3.22715144505387e-07,    # Mars
               0.000954791938424327,    # Jupiter
               0.000285885980666131,    # Saturn
               4.36625166911354e-05,    # Uranus
               5.15138902046612e-05,    # Neptune
               7.40740740830878e-09,    # Pluto (bear with me)
               ]

    ps.rx = [-0.00712377699396443,
             -0.134041839403259,
             -0.726062021587661,
             -0.189566509161968,
             1.38407082168322,
             3.97896476349658,
             6.37017339500169,
             14.502078723897,
             16.9343722652737,
             -9.87695317832301,
             ]
    ps.ry = [-0.0028353898563048766,
             -0.450586209684483,
             -0.039665992657221004,
             0.963413224416489,
             -0.008542637084489567,
             2.95779610056192,
             6.60411921341949,
             -13.6574724004636,
             -24.904781203303,
             -28.0623724768707,
             ]
    ps.rz = [-4.636758252946855e-06,
             -0.0317327781784651,
             0.0218575042921458,
             0.00338899408041853,
             0.00195338624778221,
             0.0277878676318589,
             -0.146003410253912,
             0.0266710693728891,
             0.360719859921208,
             5.35614463977912,
             ]

    ps.vx = [0.0003150431297530473,
             1.24765113964416,
             0.0537323647367395,
             -0.998489725622231,
             0.0339697602312997,
             -0.267315308034173,
             -0.2504985023359,
             0.155097409556616,
             0.149752685968701,
             0.177878490988979,
             ]
    ps.vy = [-0.0004287885979094824,
             -0.367348560236518,
             -1.17961128266637,
             -0.189356029048893,
             0.882336954083445,
             0.372606717609675,
             0.224300955779407,
             0.155754862016402,
             0.103693951061509,
             -0.0885170840283217,
             ]
    ps.vz = [-8.535355767702574e-07,
             -0.11505457416379,
             -0.0273933407210185,
             -0.0277905056358035,
             0.0258975097250254,
             0.000533607251937168,
             0.00131013765082432,
             0.00394092485741823,
             -0.000776459933629997,
             -0.0375021230229398,
             ]

    ps.id = range(ps.n)
    return ps


########## end of file ##########
