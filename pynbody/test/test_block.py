#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import time

if __name__ == "__main__":
    from pynbody.models import Plummer
    from pynbody.particles import (Particles, BlackHoles)
    from pynbody.integrator.block import Block

    numBodies = 32
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



    block = Block(0.015625, particles)

    t0 = time.time()
    block.step()
#    block.step()
    block.print_block()
    elapsed = time.time() - t0
    print('step: ', elapsed)






########## end of file ##########
