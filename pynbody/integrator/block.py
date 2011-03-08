#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import math
import numpy as np
from pynbody.particles import Particles
from pynbody.lib.decorators import selftimer
from pprint import pprint


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
    def force(self, all_particles):
        """

        """
#        print(indent*(self.level-MIN_LEVEL), 'F'+str(self.level))

        for obj0 in self.particles.itervalues():
            if obj0:
                prev_acc = obj0.acc.copy()
                mid_acc = np.zeros_like(obj0.acc)
                prev_step_density = obj0.curr_step_density.copy()
                mid_step_density = np.zeros_like(obj0.curr_step_density)
                for obj1 in all_particles.itervalues():
                    if obj1:
                        obj0.calc_acc(obj1)
                        mid_acc += obj0.acc
                        mid_step_density += obj0.curr_step_density
                obj0.acc = 2*mid_acc - prev_acc

                obj0.curr_step_density = (mid_step_density**2) / prev_step_density
                obj0.next_step_density = (obj0.curr_step_density**2) / mid_step_density

#                print('level: {0}, len: {1}'.format(self.level, len(obj0)))








class BlockStep2(object):
    """

    """
    def __init__(self, eta, particles, time=0.0,
                 max_tstep=2.0**(-MIN_LEVEL),
                 min_tstep=2.0**(-MAX_LEVEL)):
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
    def calc_block_level(self, obj):
        """

        """
        # Discretizes time steps in power-of-two
        real_tstep = self.eta / obj.next_step_density
        power = (np.log2(real_tstep) - 1).astype(np.int)
        block_tstep = 2.0**power

        # Clamp block_tstep to range given by min_tstep, max_tstep.
        block_tstep[np.where(block_tstep < self.min_tstep)] = self.min_tstep
        block_tstep[np.where(block_tstep > self.max_tstep)] = self.max_tstep

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
                block_level = self.calc_block_level(obj)
                for level in allowed_levels:
                    has_level = (level == block_level)
                    if has_level.any():
                        obj_at_level = obj[np.where(has_level)]
                        particles_at_level = Particles()
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

        particles = Particles()
        if interpolated_at_time:
            def interpolate(dt, block):
                particle = Particles()
                for (key, obj) in block.particles.iteritems():
                    if obj:
                        particle[key] = obj[:]
                        if dt < 0:
                            aux_vel = obj.vel - block.tstep * obj.acc
                        if dt > 0:
                            aux_vel = obj.vel + block.tstep * obj.acc
                        particle[key].pos += dt * aux_vel
                        particle[key].vel += dt * obj.acc
                return particle
            for block in block_list:
                dt = (interpolated_at_time - block.time)
                if dt != 0:
                    particles.append(interpolate(dt, block))
                else:
                    particles.append(block.particles)
        else:
            for block in block_list:
                particles.append(block.particles)

        # if sorting of objects by index is desired
        if sorting_by_index:
            for (key, obj) in particles.iteritems():
                if obj:
                    array = np.sort(obj.to_cmpd_struct(), order=['index'])
                    particles[key] = obj.__class__()
                    particles[key].from_cmpd_struct(array)
        return particles


    @selftimer
    def up_level(self, block, block_list):
        """

        """
#        print(indent*(block.level-MIN_LEVEL), 'Up from '+str(block.level)+' to '+str(block.upper_level), '  -->', block.time)

        for (key, obj) in block.particles.iteritems():
            if obj:
                block_level = self.calc_block_level(obj)
                where_gt_level = np.where(block_level > block.level)
                block_level[where_gt_level] = block.upper_level

#                where_lt_level = np.where(block_level < block.level)
#                block_level[where_lt_level] = block.lower_level

                if block.upper_level in block_level:
                    at_level = (block_level == block.upper_level)
                    obj_at_level = obj[np.where(at_level)]
                    particles_at_level = Particles()
                    particles_at_level.set_members(obj_at_level)
                    exist, index = self.is_there(block.upper_level, block_list)
                    if exist:
                        block_list[index].particles.append(particles_at_level)
                    else:
                        new_block = Block(block.upper_level, block.time, particles_at_level)
                        block_list.append(new_block)
                    obj_list = obj.to_cmpd_struct()[np.where(~at_level)]
                    block.particles[key] = obj.__class__()
                    block.particles[key].from_cmpd_struct(obj_list)
        return block_list


    @selftimer
    def down_level(self, block, block_list):
        """

        """
#        print(indent*(block.lower_level-MIN_LEVEL), 'Down from '+str(block.level)+' to '+str(block.lower_level))

        for (key, obj) in block.particles.iteritems():
            if obj:
                block_level = self.calc_block_level(obj)
                where_lt_level = np.where(block_level < block.level)
                block_level[where_lt_level] = block.lower_level

#                where_gt_level = np.where(block_level > block.level)
#                block_level[where_gt_level] = block.upper_level

                if block.lower_level in block_level:
                    at_level = (block_level == block.lower_level)
                    obj_at_level = obj[np.where(at_level)]
                    particles_at_level = Particles()
                    particles_at_level.set_members(obj_at_level)
                    exist, index = self.is_there(block.lower_level, block_list)
                    if exist:
                        block_list[index].particles.append(particles_at_level)
                    else:
                        new_block = Block(block.lower_level, block.time, particles_at_level)
                        block_list.append(new_block)
                    obj_list = obj.to_cmpd_struct()[np.where(~at_level)]
                    block.particles[key] = obj.__class__()
                    block.particles[key].from_cmpd_struct(obj_list)
        return block_list



    @selftimer
    def update_blocks(self, nextidx, block, block_list):
        """

        """
        new_block_list = self.up_level(block, block_list[:])

        if (nextidx < len(block_list)):
            nextblock = block_list[nextidx]
            new_block_list = self.down_level(nextblock, new_block_list)

        return self.sorted_by_level(self.remove_empty_blocks(new_block_list))




    # TODO:
    def step(self, idx=0):
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
                self.step(nextidx)
                if nextblock.time == block.time:
                    break

        block.force(self.gather(interpolated_at_time=block.time))
#        block.force(self.gather())

        if (nextidx < len(self.block_list)):
            while True:
                nextblock = self.block_list[nextidx]
                self.step(nextidx)
                if nextblock.time == block.time+block.tstep:
                    break

        block.kick()
        block.drift()
        block.time += block.tstep

        if idx == 0:
            self.time = +block.time


        self.block_list = self.update_blocks(nextidx, block, self.block_list)














class BlockStep(object):
    """

    """

    def __init__(self, eta, particles, max_tstep=2**(-4), min_tstep=2**(-24)):
        self.eta = eta
        self.time = 0.0
        self.max_tstep = max_tstep
        self.min_tstep = min_tstep
        self.block = self.scatter_on_block_levels(particles)
        self.block_depth = len(self.block)


    def print_block(self):
        pprint(self.block)
        print('block_depth:', self.block_depth)


    def calc_block_level(self, obj, level=None):
        """

        """
        # Discretizes time steps in power-of-two
        tstep = self.eta / obj.next_step_density
#        tstep = self.eta / obj.step_density
        power = (np.log2(tstep) - 1).astype(np.int)
        block_tstep = 2.0**power

        # Clamp block_tstep to range given by min_tstep, max_tstep.
        where_lt_min = np.where(block_tstep < self.min_tstep)
        block_tstep[where_lt_min] = self.min_tstep
        where_gt_max = np.where(block_tstep > self.max_tstep)
        block_tstep[where_gt_max] = self.max_tstep

        # Converts from block_tstep to block_level
        block_level = -np.log2(block_tstep).astype(np.int)

        if level:
            block_level[np.where(block_level < level)] = level-1
            block_level[np.where(block_level > level)] = level+1

#        max_level = np.max(block_level)
#        min_level = np.min(block_level)
#        num_levels = (max_level - min_level) + 1
#        print(block_level)
#        print(block_tstep)
#        print(max_level, min_level, num_levels)
#        if max_level > 6: print(('max_level: '+str(max_level)+' ')*20)

        return block_level


    def interpolate(self, particles, at_time):
        """

        """
        interpolated = Particles()
        for (key, obj) in particles.items():
            if obj:
                interpolated[key] = obj[:]
                interpolated[key].vel += ((at_time - obj.time) * obj.acc.T).T
                aux_vel = obj.vel - (obj.tstep * obj.acc.T).T
                interpolated[key].pos += ((at_time - obj.time) * aux_vel.T).T
        return interpolated


    def gather_from_block_levels(self, block=None, sorting=False):
        """

        """
        if block is None:
            block = self.block
        particles = Particles()
        for (level, particles_at_level) in block:
            for (key, obj) in particles_at_level.items():
                if particles[key]:
                    array = np.append(particles[key].to_cmpd_struct(),
                                      obj.to_cmpd_struct())
                    particles[key] = obj.__class__()
                    particles[key].from_cmpd_struct(array)
                else:
                    particles[key] = obj

        # if sorting of objects by index is desired
        if sorting:
            for (key, obj) in particles.items():
                if particles[key]:
                    array = np.sort(obj.to_cmpd_struct(), order=['index'])
                    particles[key] = obj.__class__()
                    particles[key].from_cmpd_struct(array)
        return particles


    def scatter_on_block_levels(self, particles):
        """

        """
        block = []
        allowed_levels = range(-np.log2(self.max_tstep).astype(np.int),
                               1-np.log2(self.min_tstep).astype(np.int))
        for (key, obj) in particles.items():
            if obj:
                block_level = self.calc_block_level(obj)
                for level in allowed_levels:
                    has_level = block_level == level
                    if has_level.any():
                        tmp = obj[:]
                        tmp.tstep = 2.0**(-block_level)
                        obj_at_level = tmp[np.where(has_level)]
                        particles_at_level = Particles()
                        particles_at_level.set_members(obj_at_level)
                        if block:
                            exist_level = False
                            level_index = None
                            for item in block:
                                if level in item:
                                    exist_level = True
                                    level_index = block.index(item)
                            if exist_level:
                                block[level_index][1][key] = obj_at_level
                            else:
                                block.append((level, particles_at_level))
                        else:
                            block.append((level, particles_at_level))
        return sorted(block)


    def drift(self, block):
        """

        """
        level, obj = block

        print(indent*(level-4), 'D'+str(level)+':', 2.0**(-level))

        if obj['body']:
            tstep = 2.0**(-level)
            obj['body'].time += tstep  # obj['body'].tstep
            obj['body'].drift(tstep)




    def kick(self, block):
        """

        """
        level, obj = block

        print(indent*(level-4), 'K'+str(level)+':', 2.0**(-level))

        if obj['body']:
            tstep = 2.0**(-level)
            obj['body'].kick(tstep)



    def force(self, block):
        """

        """
        level, obj = block

        print(indent*(level-4), 'F'+str(level))

        if obj['body']:
            body = obj['body']


            rho_prev = +body.curr_step_density
            prev_acc = +body.acc

            particles = self.gather_from_block_levels()
#            particles = self.interpolate(particles, body.time[0])
            body.calc_acc(particles['body'])

            body.acc = 2*body.acc - prev_acc

            rho_mid = +body.curr_step_density


            body.curr_step_density = (rho_mid**2) / rho_prev
            body.next_step_density = (body.curr_step_density**2) / rho_mid

            if (body.next_step_density < 0).any():
                print(' less than zero'*250)
#            print(prev_step_density)
#            print(body.step_density)

#            print(rho_prev)
#            print(rho_mid)
#            print(body.step_density)
#            print(body.step_density_next)


    def has_level(self, level, block):
        has = False
        index = None
        for item in block:
            if level in item:
                has = True
                index = block.index(item)
        return (has, index)


    def remove_empty_levels(self):
        block = self.block[:]
        for item in block:
            level, particles = item
            has_particles = False
            for (key, obj) in particles.items():
                if obj:
                    has_particles = True
            if not has_particles:
#                if item == self.block[-1]:
                block.remove(item)
        self.block = sorted(block)
        self.block_depth = len(self.block)


    def change_level(self, idx, offset):
        """

        """
        level0, particles0 = self.block[idx]
        next_level = level0 + offset
        for (key, obj) in particles0.items():
            if obj:
                block_level = self.calc_block_level(obj, level0)
                if next_level in block_level:
                    has_level, level_index = self.has_level(next_level, self.block)
                    tmp = obj[:]
                    tmp.tstep = 2.0**(-block_level)
                    at_level = (block_level == next_level)
                    obj_at_level = tmp[np.where(at_level)]
                    particles_at_level = Particles()
                    particles_at_level.set_members(obj_at_level)
                    if has_level:
                        level1, particles1 = self.block[level_index]
                        if particles1[key]:
                            arr = np.append(particles1[key].to_cmpd_struct(),
                                            obj_at_level.to_cmpd_struct())
                            particles1[key] = obj.__class__()
                            particles1[key].from_cmpd_struct(arr)
                        else:
                            particles1[key] = obj_at_level
                    else:
                        self.block.append((next_level, particles_at_level))
                    obj_list = tmp.to_cmpd_struct()[np.where(~at_level)]
                    particles0[key] = obj.__class__()
                    particles0[key].from_cmpd_struct(obj_list)
        self.remove_empty_levels()



    def up_level(self, idx):
        level0, particles0 = self.block[:][idx]
        self.change_level(idx, +1)
        print(indent*(level0-4),
              'Up from '+str(level0)+' to '+str(level0+1))



    def down_level(self, idx):
        level0, particles0 = self.block[:][idx]
        self.change_level(idx, -1)
        print(indent*(level0-1-4),
              'Down from '+str(level0)+' to '+str(level0-1))



    def step(self, idx=0):
        """

        """
        nextidx = idx + 1

        self.drift(self.block[idx])
        self.kick(self.block[idx])

        if (nextidx < self.block_depth):
            for i in range(2**(self.block[nextidx][0]-self.block[idx][0]-1)):
                self.step(nextidx)
#                self.up_level(nextidx)
#                if (nextidx+1 < self.block_depth):
#                    self.down_level(nextidx+1)

        self.force(self.block[idx])

        if (nextidx < self.block_depth):
            for i in range(2**(self.block[nextidx][0]-self.block[idx][0]-1)):
                self.step(nextidx)
#                self.up_level(nextidx)
#                if (nextidx+1 < self.block_depth):
#                    self.down_level(nextidx+1)

        self.kick(self.block[idx])
        self.drift(self.block[idx])

#        for item in self.block:
#            print(str(item[0])+':', len(item[1]['body']))

#        if (idx == 0):

        self.up_level(idx)
        if (nextidx < self.block_depth):
            self.down_level(nextidx)

#        self.up_level(idx)

#        for item in self.block:
#            print(str(item[0])+':', len(item[1]['body']))

        if idx == 0:
            self.time += 2*2.0**(-self.block[0][0])



########## end of file ##########
