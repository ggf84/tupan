#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import numpy as np
from pprint import pprint
from pynbody.lib.decorators import selftimer


indent = ' '*4

MIN_LEVEL = 4
MAX_LEVEL = 34


class Block(object):
    """

    """
    def __init__(self, level, time, particles):
        self.level = level
        self.time = time
        self.particles = particles
        self.tstep = 2.0**(-level)
        self.lower_level = level-1
        self.upper_level = level+1

    def __repr__(self):
        fmt = 'Block({0}, {1}, {2}, {3})'
        return fmt.format(self.level, self.tstep, self.time, self.particles)


    @selftimer
    def drift(self):
        """

        """
#        print(indent*(self.level-MIN_LEVEL), 'D'+str(self.level)+':', self.tstep)

        for obj in self.particles.itervalues():
            if obj:
                obj.drift(self.tstep)


    @selftimer
    def kick(self):
        """

        """
#        print(indent*(self.level-MIN_LEVEL), 'K'+str(self.level)+':', self.tstep)

        for obj in self.particles.itervalues():
            if obj:
                obj.kick(self.tstep)


    @selftimer
    def interpolated(self, at_time):
        """

        """
        dt = at_time - self.time
        if dt != 0:
            btstep = -self.tstep if dt < 0 else self.tstep
            particles = self.particles.copy()
            for obj in particles.itervalues():
                if obj:
                    aux_acc = obj.acc
                    aux_vel = obj.vel #+ btstep * aux_acc
                    obj.pos += dt * aux_vel
                    obj.vel += dt * aux_acc
            return particles
        return self.particles


    @selftimer
    def force(self, all_particles):
        """

        """
#        print(indent*(self.level-MIN_LEVEL), 'F'+str(self.level))

        for obj0 in self.particles.itervalues():
            if obj0:
                prev_acc = obj0.acc.copy()
                mid_acc = np.zeros_like(obj0.acc)
                prev_stepdens = obj0.stepdens[:,0].copy()
                mid_stepdens = np.zeros_like(obj0.stepdens[:,0])
                for obj1 in all_particles.itervalues():
                    if obj1:
                        obj0.set_acc(obj1)
                        mid_acc += obj0.acc
                        mid_stepdens += obj0.stepdens[:,0]
                obj0.acc[:] = 2*mid_acc - prev_acc
                obj0.stepdens[:,0] = (mid_stepdens**2) / prev_stepdens
                obj0.stepdens[:,1] = (obj0.stepdens[:,0]**2) / mid_stepdens

#                print('level: {0}, len: {1}'.format(self.level, len(obj0)))








class BlockStep(object):
    """

    """
    def __init__(self, time, eta, particles,
                 max_tstep=2.0**(-MIN_LEVEL),
                 min_tstep=2.0**(-MAX_LEVEL)):
        self.ParticlesClass = particles.__class__
        self.eta = eta
        self.time = time
        self.max_tstep = max_tstep
        self.min_tstep = min_tstep
        self.block_list = self.scatter(particles)

    @selftimer
    def print_block(self):
        pprint(self.block_list)
        print('block_list length:', len(self.block_list))

    @selftimer
    def print_levels(self, block_list=None):
        if block_list is None:
            block_list = self.block_list
        levels = []
        for block in block_list:
            levels.append(block.level)
        print(levels)

    @selftimer
    def sorted_by_level(self, block_list):
        def cmp(x, y):
            if x.level < y.level: return -1
            if x.level > y.level: return +1
            return 0
        return sorted(block_list, cmp=cmp)

    @selftimer
    def is_there(self, level, block_list):
        has_level = False
        index = None
        for block in block_list:
            if level == block.level:
                has_level = True
                index = block_list.index(block)
        return (has_level, index)

    @selftimer
    def remove_empty_blocks(self, block_list):
        for block in block_list:
            if not block.particles.any():
                block_list.remove(block)
        return block_list

    @selftimer
    def calc_block_level(self, currtime, obj):
#    def calc_block_level(self, obj):
        """

        """
        # Discretizes time steps in power-of-two
        real_tstep = self.eta / obj.stepdens[:,1]
        power = (np.log2(real_tstep) - 1).astype(np.int)
        block_tstep = 2.0**power

        # Clamp block_tstep to range given by min_tstep, max_tstep.
        block_tstep = block_tstep.clip(self.min_tstep, self.max_tstep)

        mod = (currtime+2*block_tstep)%(2*block_tstep)
        is_ne_zero = (mod != 0)
        while is_ne_zero.any():
            block_tstep[np.where(is_ne_zero)] /= 2
            mod = (currtime+2*block_tstep)%(2*block_tstep)
            is_ne_zero = (mod != 0)

        # Converts from block_tstep to block_level and returns
        return -np.log2(block_tstep).astype(np.int)

    @selftimer
    def scatter(self, particles):
        """

        """
        block_list = []
        allowed_levels = range(-np.log2(self.max_tstep).astype(np.int),
                               1-np.log2(self.min_tstep).astype(np.int))
        for (key, obj) in particles.iteritems():
            if obj:
                block_level = self.calc_block_level(self.time, obj)
#                block_level = self.calc_block_level(obj)
                for level in allowed_levels:
                    has_level = (level == block_level)
                    if has_level.any():
                        obj_at_level = obj[np.where(has_level)]
                        particles_at_level = self.ParticlesClass()
                        particles_at_level.set_members(obj_at_level)
                        block = Block(level, 0.0, particles_at_level)
                        if block_list:
                            exist, index = self.is_there(level, block_list)
                            if exist:
                                block_list[index].particles[key] = obj_at_level
                            else:
                                block_list.append(block)
                        else:
                            block_list.append(block)
        return self.sorted_by_level(block_list)

    @selftimer
    def gather(self, block_list=None,
               interpolated_at_time=None,
               sorting_by_index=False):
        """

        """
        if block_list is None:
            block_list = self.block_list

        particles = self.ParticlesClass()
        if interpolated_at_time:
            for block in block_list:
                pinterp = block.interpolated(interpolated_at_time)
                particles.append(pinterp)
        else:
            for block in block_list:
                particles.append(block.particles)

        # if sorting of objects by index is desired
        if sorting_by_index:
            for (key, obj) in particles.iteritems():
                if obj:
                    array = np.sort(obj.get_data(), order=['index'])
                    particles[key] = obj.__class__()
                    particles[key].set_data(array)
        return particles


    @selftimer
    def up_level(self, block, block_list):
        """

        """
#        print(indent*(block.level-MIN_LEVEL), 'Up from '+str(block.level)+' to '+str(block.upper_level), '  -->', block.time)

        for (key, obj) in block.particles.iteritems():
            if obj:
                block_level = self.calc_block_level(block.time, obj)
#                block_level = self.calc_block_level(obj)

                where_gt_level = np.where(block_level > block.level)
                block_level[where_gt_level] = block.upper_level

#                where_lt_level = np.where(block_level < block.level)
#                block_level[where_lt_level] = block.lower_level

                if block.upper_level in block_level:
                    at_level = (block_level == block.upper_level)
                    obj_at_level = obj[np.where(at_level)]
                    particles_at_level = self.ParticlesClass()
                    particles_at_level.set_members(obj_at_level)
                    exist, index = self.is_there(block.upper_level, block_list)
                    if exist:
                        block_list[index].particles.append(particles_at_level)
                    else:
                        new_block = Block(block.upper_level, block.time, particles_at_level)
                        block_list.append(new_block)
                    obj_list = obj.get_data()[np.where(~at_level)]
                    block.particles[key] = obj.__class__()
                    block.particles[key].set_data(obj_list)
        return block_list


    @selftimer
    def down_level(self, block, block_list):
        """

        """
#        print(indent*(block.lower_level-MIN_LEVEL), 'Down from '+str(block.level)+' to '+str(block.lower_level))

        for (key, obj) in block.particles.iteritems():
            if obj:
                block_level = self.calc_block_level(block.time, obj)
#                block_level = self.calc_block_level(obj)

                where_lt_level = np.where(block_level < block.level)
                block_level[where_lt_level] = block.lower_level

#                where_gt_level = np.where(block_level > block.level)
#                block_level[where_gt_level] = block.upper_level

                if block.lower_level in block_level:
                    at_level = (block_level == block.lower_level)
                    obj_at_level = obj[np.where(at_level)]
                    particles_at_level = self.ParticlesClass()
                    particles_at_level.set_members(obj_at_level)
                    exist, index = self.is_there(block.lower_level, block_list)
                    if exist:
                        block_list[index].particles.append(particles_at_level)
                    else:
                        new_block = Block(block.lower_level, block.time, particles_at_level)
                        block_list.append(new_block)
                    obj_list = obj.get_data()[np.where(~at_level)]
                    block.particles[key] = obj.__class__()
                    block.particles[key].set_data(obj_list)
        return block_list



    @selftimer
    def update_blocks(self, nextidx, block, block_list):
        """

        """
        new_block_list = self.up_level(block, block_list[:])

        if (nextidx < len(new_block_list)):
            nextblock = new_block_list[nextidx]
#        if (nextidx < len(block_list)):
#            nextblock = block_list[nextidx]
            new_block_list = self.down_level(nextblock, new_block_list[:])


        new_block_list = self.remove_empty_blocks(new_block_list)
        new_block_list = self.sorted_by_level(new_block_list)
        return new_block_list



    # TODO:
    def recursive_step(self, idx=0):
        """

        """
        nextidx = idx + 1
        block = self.block_list[idx]

        block.time += block.tstep
        block.drift()
        block.kick()

        if (nextidx < len(self.block_list)):
            while True:
                nextblock = self.block_list[nextidx]
                self.recursive_step(nextidx)
                if nextblock.time == block.time:
                    break

        block.force(self.gather(interpolated_at_time=block.time))
#        block.force(self.gather())

        if (nextidx < len(self.block_list)):
            while True:
                nextblock = self.block_list[nextidx]
                self.recursive_step(nextidx)
                if nextblock.time == block.time+block.tstep:
                    break

        block.kick()
        block.drift()
        block.time += block.tstep

        if idx == 0:
            self.time = +block.time

        self.block_list = self.update_blocks(nextidx, block, self.block_list)









    def recursive_step2(self, idx=0):
        """

        """
        nextidx = idx + 1
        block = self.block_list[idx]

        if (nextidx < len(self.block_list)):
            while True:
                nextblock = self.block_list[nextidx]
                self.recursive_step(nextidx)
                if nextblock.time == block.time+block.tstep:
                    break

        block.time += block.tstep
        block.drift()
        block.kick()

        block.force(self.gather(interpolated_at_time=block.time))
#        block.force(self.gather())

        block.kick()
        block.drift()
        block.time += block.tstep

        if (nextidx < len(self.block_list)):
            while True:
                nextblock = self.block_list[nextidx]
                self.recursive_step(nextidx)
                if nextblock.time == block.time:
                    break

        if idx == 0:
            self.time = +block.time

        self.block_list = self.update_blocks(nextidx, block, self.block_list)




    @selftimer
    def step(self):
        self.recursive_step2()




########## end of file ##########
