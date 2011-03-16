#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import pylab
import time
import numpy as np
from pynbody.integrator import LeapFrog, BlockStep


def plot_coords(x, y, fname):
    pylab.plot(x, y, ',')
    pylab.savefig(fname)
#    pylab.show()


if __name__ == "__main__":
    from pynbody.models import Plummer
    from pynbody.particles import (Particles, BlackHole)


    numBodies = 256
    p = Plummer(numBodies, seed=1)
    p.make_plummer()
    p.bodies.set_acc(p.bodies)
    particles = Particles()
    particles.set_members(p.bodies)


#    # BlackHole
#    p = Plummer(3, seed=1)
#    p.make_plummer()
#    p.bodies.calc_acc(p.bodies)

#    bhdata = BlackHole()
#    bhdata.fromlist([tuple(b)+([0.0, 0.0, 0.0],) for b in p.bodies])

#    particles.set_members(bhdata)




#    from pynbody.io import HDF5IO
#    io = HDF5IO('input4block.hdf5')
#    particles = io.read_snapshot()



#    particles['body'].eps2.fill(0.0015625)
    particles['body'].eps2.fill(0.5*((4.0/numBodies)**2))
#    particles['body'].vel.fill(0.0)
    particles['body'].set_phi(particles['body'])
    particles['body'].set_acc(particles['body'])
    particles['body'].stepdens[:,1] = particles['body'].stepdens[:,0].copy()


    x = particles['body'].pos[:,0]
    y = particles['body'].pos[:,1]
    plot_coords(x, y, './png/p0.png')


    e0 = particles['body'].get_ekin()+particles['body'].get_epot()
    com0 = particles['body'].get_Rcenter_of_mass()
    linmom0 = particles['body'].get_linear_mom()
    angmom0 = particles['body'].get_angular_mom()



#    leapFrog = LeapFrog(0.0, 2.0**(-7), particles)
#    i = 0
#    while leapFrog.time < 64.0:
#        i += 1
#        tout = +leapFrog.time
#        while leapFrog.time-tout < 0.03125:
#            leapFrog.step()
#        p = leapFrog.particles.copy()
##        x = p['body'].pos[:,0]
##        y = p['body'].pos[:,1]
##        plot_coords(x, y, './png/p'+str(i)+'.png')
#        p['body'].set_phi(p['body'])
#        e1 = p['body'].get_ekin()+p['body'].get_epot()
#        linmom1 = p['body'].get_linear_mom()
#        angmom1 = p['body'].get_angular_mom()

#        plot_coords(leapFrog.time, (e1-e0)/e0, './png/p'+str(i)+'.png')

#        print('{0}: {1} {2} {3}\n{4}\n{5}'.format(leapFrog.time, e0, e1, (e1-e0)/e0, linmom1-linmom0, angmom1-angmom0))
#    p = leapFrog.particles.copy()
#    p['body'].set_phi(p.copy()['body'])
#    e1 = p['body'].get_ekin()+p['body'].get_epot()
#    linmom1 = p['body'].get_linear_mom()
#    angmom1 = p['body'].get_angular_mom()
#    print('{0}: {1} {2} {3}\n{4}\n{5}'.format(leapFrog.time, e0, e1, (e1-e0)/e0, linmom1-linmom0, angmom1-angmom0))





    block = BlockStep(1.0/128, particles)

#    block.print_block()
#    print(block.gather().copy())

    i = 0
    while block.time < 4.0:
        i += 1
        tout = +block.time
        while block.time-tout < 0.03125:
            block.step()
        p = block.gather().copy()
        x = p['body'].pos[:,0]
        y = p['body'].pos[:,1]
        plot_coords(x, y, './png/p'+str(i)+'.png')
        p['body'].set_phi(p['body'])
        e1 = p['body'].get_ekin()+p['body'].get_epot()
        com1 = p['body'].get_Rcenter_of_mass()
        linmom1 = p['body'].get_linear_mom()
        angmom1 = p['body'].get_angular_mom()

#        plot_coords(block.time, (e1-e0)/e0, './png/p'+str(i)+'.png')

        print('{0}: {1} {2} {3}\n{4}\n{5}\n{6}'.format(block.time, e0, e1, (e1-e0)/e0, com1-com0, linmom1-linmom0, angmom1-angmom0))
    p = block.gather().copy()
    p['body'].set_phi(p.copy()['body'])
    e1 = p['body'].get_ekin()+p['body'].get_epot()
    com1 = p['body'].get_Rcenter_of_mass()
    linmom1 = p['body'].get_linear_mom()
    angmom1 = p['body'].get_angular_mom()
    print('{0}: {1} {2} {3}\n{4}\n{5}\n{6}'.format(block.time, e0, e1, (e1-e0)/e0, com1-com0, linmom1-linmom0, angmom1-angmom0))





########## end of file ##########
