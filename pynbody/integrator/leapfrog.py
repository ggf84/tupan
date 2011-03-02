#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import numpy as np


class LeapFrog(object):
    """

    """
    def __init__(self, time, tstep, particles):
        self.time = time
        self.tstep = tstep
        self.particles = particles


    def drift(self):
        """

        """
        for (key, obj) in self.particles.items():
            if obj:
                obj.drift(self.tstep)


    def kick(self):
        """

        """
        for (key, obj) in self.particles.items():
            if obj:
                obj.kick(self.tstep)


    def force(self, all_particles):
        """

        """
        for (key0, obj0) in self.particles.items():
            if obj0:
                prev_obj0_acc = obj0.acc.copy()
                mid_obj0_acc = np.zeros_like(obj0.acc)
#                prev_obj0_step_density = obj0.curr_step_density.copy()
#                mid_obj0_step_density = np.zeros_like(obj0.curr_step_density)
                for (key1, obj1) in all_particles.items():
                    if obj1:
                        print('len: {0}, {1}'.format(len(obj0), len(obj1)))
                        obj0.calc_acc(obj1)         # obj0's acc due to obj1
                        mid_obj0_acc += obj0.acc    # sum obj1's contribution
#                        mid_obj0_step_density += obj0.curr_step_density
                obj0.acc = 2*mid_obj0_acc - prev_obj0_acc
#                tmp = (mid_obj0_step_density**2) / prev_obj0_step_density
#                obj0.curr_step_density = tmp.copy()
#                tmp = (obj0.curr_step_density**2) / mid_obj0_step_density
#                obj0.next_step_density = tmp.copy()


    def step(self):
        self.time += self.tstep
        self.drift()
        self.kick()
        self.force(self.particles)
        self.kick()
        self.drift()
        self.time += self.tstep



########## end of file ##########
