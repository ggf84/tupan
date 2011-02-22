#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import math
import numpy as np
from pynbody.particles import Particles

from pprint import pprint
from pynbody.lib.decorators import selftimer


indent = ' '*3


class Block(object):
    """

    """

    def __init__(self, eta, particles, max_tstep=2**(-3), min_tstep=2**(-23)):
        self.eta = eta
        self.max_tstep = max_tstep
        self.min_tstep = min_tstep

        self.particles = particles
        self.block = []
        self.block_depth = 0
        self.allowed_levels = range(-np.log2(self.max_tstep).astype(np.int),
                                    1-np.log2(self.min_tstep).astype(np.int))

        self.scatter_on_block_levels(self.particles)

        self.gather_from_block_levels(6)

    def print_block(self):
        pprint(self.block)
        print('block_depth:', self.block_depth)


    def calc_block_level(self, obj):
        """

        """
        # Discretizes time steps in power-of-two
        tstep = self.eta / obj.step_density
        power = (np.log2(tstep) - 1).astype(np.int)
        block_tstep = 2.0**power

        # Clamp block_tstep to range given by min_tstep, max_tstep.
        where_lt_min = np.where(block_tstep < self.min_tstep)
        block_tstep[where_lt_min] = self.min_tstep
        where_gt_max = np.where(block_tstep > self.max_tstep)
        block_tstep[where_gt_max] = self.max_tstep

        # Converts from block_tstep to block_level
        block_level = -np.log2(block_tstep).astype(np.int)

#        max_level = np.max(block_level)
#        min_level = np.min(block_level)
#        num_levels = (max_level - min_level) + 1
#        print(block_level)
#        print(block_tstep)
#        print(k, max_level, min_level, num_levels)

        return block_level

    def interpolate(self, at_time, particles):
        pass

    @selftimer
    def gather_from_block_levels(self, at_level, _sorted=False):
        """

        """
        particles = Particles()
        for (level, particles_at_level) in self.block:
            for (name, obj) in particles_at_level.items():
                if particles[name]:
                    tmp_array = np.append(particles[name].to_cmpd_struct(),
                                          obj.to_cmpd_struct())
                    tmp_obj = obj.__class__()
                    tmp_obj.from_cmpd_struct(tmp_array)
                    particles[name] = tmp_obj
                else:
                    particles[name] = obj

        # if sorting of objects by index is desired
        if _sorted:
            for (name, obj) in particles.items():
                if particles[name]:
                    sorted_array = np.sort(obj.to_cmpd_struct(), order=['index'])
                    tmp_obj = obj.__class__()
                    tmp_obj.from_cmpd_struct(sorted_array)
                    particles[name] = tmp_obj
        return particles

    def scatter_on_block_levels(self, particles):
        """

        """
        for (name, obj) in particles.items():
            if obj:
                block_level = self.calc_block_level(obj)
                for level in self.allowed_levels:
                    has_level = block_level == level
                    if has_level.any():
                        obj_list = obj.to_cmpd_struct()[np.where(has_level)]
                        obj_at_level = obj.__class__()
                        obj_at_level.from_cmpd_struct(obj_list)
                        particles_at_level = Particles()
                        particles_at_level.set_members(obj_at_level)
                        if self.block:
                            exist_level = False
                            level_index = None
                            for item in self.block:
                                if level == item[0]:
                                    exist_level = True
                                    level_index = self.block.index(item)
                            if exist_level:
                                self.block[level_index][1][name] = obj_at_level
                            else:
                                self.block.append((level, particles_at_level))
                        else:
                            self.block.append((level, particles_at_level))
        self.block = sorted(self.block)
        self.block_depth = len(self.block)


    def drift(self, block):
        """

        """
        level, obj = block

        print(indent*level, 'D'+str(level)+':', 2.0**(-level))

        if obj['body']:
            obj['body'].drift(2.0**(-level))




    def kick(self, block):
        """

        """
        level, obj = block

        print(indent*level, 'K'+str(level)+':', 2.0**(-level))

        if obj['body']:
            obj['body'].kick(2.0**(-level))



    def force(self, block):
        """

        """
        level, obj = block

#        print(indent*level, 'F'+str(level))

        if obj['body']:
            body = obj['body']

#            print(body[0])

            body.time += 2.0**(-level+1)

            print(indent*level, 'F'+str(level), body.time[0])

            prev_step_density = +body.step_density
            prev_acc = +body.acc

            particles = self.gather_from_block_levels(0)
            body.calc_acc(particles['body'])
#            body.calc_acc(body)

            mid_step_density = +body.step_density
            mid_acc = +body.acc
            body.acc += body.acc - prev_acc
            body.step_density += body.step_density - prev_step_density

#            body.time += 2.0**(-level)


            if (body.step_density < 0).any():
                print(' less than zero'*1000)
#            print(prev_step_density)
#            print(body.step_density)


#            print(self.particles['body'][body[0].index])
#            print(body[0])

            print('---'*10)
            print(prev_acc[0], prev_step_density[0])
            print('---')
            print(mid_acc[0], mid_step_density[0])
            print('---')
            print(body.acc[0], body.step_density[0])


    def step(self, idx=0):
        """

        """
        nextidx = idx + 1

        self.drift(self.block[idx])
        self.kick(self.block[idx])

        if (nextidx < self.block_depth):
            self.step(nextidx)

        self.force(self.block[idx])

        if (nextidx < self.block_depth):
            self.step(nextidx)

        self.kick(self.block[idx])
        self.drift(self.block[idx])


########## end of file ##########
