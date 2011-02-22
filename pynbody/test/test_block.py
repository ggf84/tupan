#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import time

if __name__ == "__main__":
    from pynbody.models import Plummer
    from pynbody.particles import (Particles, BlackHoles)
    from pynbody.integrator.block import Block

    numBodies = 4096
    p = Plummer(numBodies, seed=1)
    p.make_plummer()
    p.bodies.calc_acc(p.bodies)
    particles = Particles()
    particles.set_members(p.bodies)


    # BlackHoles
    p = Plummer(3, seed=1)
    p.make_plummer()
    p.bodies.calc_acc(p.bodies)

    bhdata = BlackHoles()
    bhdata.fromlist([tuple(b)+([0.0, 0.0, 0.0],) for b in p.bodies])

    particles.set_members(bhdata)

#-0.264511001275

    block = Block(0.015625, particles)

    t0 = time.time()
#    block.step()
    for i in range(4):
        block.step()
    block.print_block()
    elapsed = time.time() - t0
    print('step: ', elapsed)


    particles2 = block.gather_from_block_levels(0)
    print(particles2['body'].time)

    print(particles['body'].get_ekin()+particles['body'].get_epot())
    print(particles2['body'].get_ekin()+particles2['body'].get_epot())


########## end of file ##########
