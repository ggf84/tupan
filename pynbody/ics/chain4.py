#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from pynbody.particles import Particles


def make_system(particles_as_type='body', **kwargs):
    _type = particles_as_type
    particles = Particles({_type: 4})
    particles[_type].id = [0, 1, 2, 3]
    particles[_type].mass = [1.0, 1.0, 1.0, 1.0]
    particles[_type].eps2 = [0.0, 0.0, 0.0, 0.0]
    particles[_type].pos = [
                             [ 1.382857,-0.0     , 0.0],
                             [ 0.0     , 0.157030, 0.0],
                             [-1.382857, 0.0     , 0.0],
                             [ 0.0     ,-0.157030, 0.0],
                           ]
    particles[_type].vel = [
                             [ 0.0     , 0.584873, 0.0],
                             [ 1.871935, 0.0     , 0.0],
                             [ 0.0     ,-0.584873, 0.0],
                             [-1.871935, 0.0     , 0.0],
                           ]
    if _type == 'blackhole':
        spin = [
                 [0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0],
               ]
        particles[_type].spin = kwargs.pop("spin", spin)

    return particles


########## end of file ##########
