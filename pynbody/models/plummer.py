#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from __future__ import print_function
import sys
import random
import math

from ggf84decor import selftimer
from pynbody.io import HDF5IO
from pynbody.particles import Particles


__all__ = ['Plummer']


def scale_mass(bodies, m_scale):
    bodies.mass *= m_scale

def scale_pos(bodies, r_scale):
    bodies.pos *= r_scale

def scale_vel(bodies, v_scale):
    bodies.vel *= v_scale

def scale_to_virial(particles, ekin, epot, etot):
    scale_vel(particles["body"], math.sqrt(-0.5*epot/ekin))
    ekin = -0.5*epot
    etot = ekin + epot
    scale_pos(particles["body"], etot/(-0.25))
    particles["body"].set_phi(particles)
    epot = particles["body"].get_total_epot()
    scale_vel(particles["body"], math.sqrt(0.5*abs(epot/etot)))


def scale_to_nbody_units(particles):
#    scale_mass(particles["body"], 1.0/particles["body"].get_mass())
#    particles["body"].set_phi(particles)

    e = particles["body"].get_total_energies()
    print(e.kin, e.pot, e.tot, e.vir)
    scale_to_virial(particles, e.kin, e.pot, e.tot)
    e = particles["body"].get_total_energies()
    print(e.kin, e.pot, e.tot, e.vir)





class Plummer(object):
    """  """

    def __init__(self, num, imf, mfrac=0.999, epsf=0.0, epstype='b', seed=None):
        self.num = num
        self.imf = imf
        self.mfrac = mfrac
        self.epsf = epsf
        self.epstype = epstype
        self.particles = Particles({"body": num})
        random.seed(seed)


    def set_eps2(self, mass):
        n = self.num

        eps_a = self.epsf * mass
        eps_b = (self.epsf / (n * (n * mass)**0.5))

        eps_a_mean = ((mass*(eps_a**2)).sum())**0.5
        eps_b_mean = ((mass*(eps_b**2)).sum())**0.5

        eps_a *= (eps_b_mean / eps_a_mean)

        if 'a' in self.epstype:
            return eps_a
        elif 'b' in self.epstype:
            return eps_b
        else:
            return 0


    def set_pos2(self, mrand):
        radius = 1.0 / math.sqrt(math.pow(mrand, -2.0/3.0) - 1.0)
        theta = math.acos(random.uniform(-1.0, 1.0))
        phi = random.uniform(0.0, 2.0 * math.pi)
        rx = radius * math.sin(theta) * math.cos(phi)
        ry = radius * math.sin(theta) * math.sin(phi)
        rz = radius * math.cos(theta)
        return (rx, ry, rz)



    def set_pos(self, irand):
        m_min = (irand * self.mfrac) / self.num
        m_max = ((irand+1) * self.mfrac) / self.num
        mrand = random.uniform(m_min, m_max)
        radius = 1.0 / math.sqrt(math.pow(mrand, -2.0/3.0) - 1.0)
        theta = math.acos(random.uniform(-1.0, 1.0))
        phi = random.uniform(0.0, 2.0 * math.pi)
        rx = radius * math.sin(theta) * math.cos(phi)
        ry = radius * math.sin(theta) * math.sin(phi)
        rz = radius * math.cos(theta)
        return (rx, ry, rz)


    def set_vel(self, pot):
        q = 1.0
        g = 1.0
        while (g > (q - 1)):
            q = random.uniform(0.0, 1.0)
            g = random.uniform(-0.75, 0.0)
        velocity = math.sqrt(2 * (q - 1) * pot)
        theta = math.acos(random.uniform(-1.0, 1.0))
        phi = random.uniform(0.0, 2.0 * math.pi)
        vx = velocity * math.sin(theta) * math.cos(phi)
        vy = velocity * math.sin(theta) * math.sin(phi)
        vz = velocity * math.cos(theta)
        return (vx, vy, vz)



#    def set_vel(self, pot):
#        q = 1.0
#        g = 2.0
#        while (g > (1 - q)):
#            q = random.uniform(-1.0, 1.0)
#            g = random.uniform(0.0, 2.0)
#        velocity = math.sqrt(-g * pot)
#        theta = math.acos(random.uniform(-1.0, 1.0))
#        phi = random.uniform(0.0, 2.0 * math.pi)
#        vx = velocity * math.sin(theta) * math.cos(phi)
#        vy = velocity * math.sin(theta) * math.sin(phi)
#        vz = velocity * math.cos(theta)
#        return (vx, vy, vz)


    @selftimer
    def set_bodies(self):
        """  """
        n = self.num
        ilist = list(range(n))

        # set index
        self.particles["body"].index[:] = ilist

        s = random.getstate()

        # set mass
        self.particles["body"].mass[:] = self.imf.sample(n)
        self.particles["body"].mass /= self.particles["body"].get_total_mass()
        self.particles["body"].update_total_mass()

        # set eps2
        self.particles["body"].eps2[:] = self.set_eps2(self.particles["body"].mass.copy())

        random.setstate(s)

        # set pos
        self.particles["body"].pos[:] = [self.set_pos(i) for i in random.sample(ilist,n)]

#        mcum = self.particles["body"].mass.cumsum()
#        mcum /= mcum.max()
#        mcum *= self.mfrac
#        self.particles["body"].pos[:] = [self.set_pos2(i) for i in random.sample(mcum, n)]

        # set phi
        self.particles["body"].set_phi(self.particles)
        # set vel
        self.particles["body"].vel[:] = [self.set_vel(p) for p in self.particles["body"].phi]


    def make_plummer(self):
        self.set_bodies()
        self.particles.reset_center_of_mass()
        scale_to_nbody_units(self.particles)
        self.particles.set_acc(self.particles)


    def write_snapshot(self, fname="plummer"):
        io = HDF5IO(fname+".hdf5", 'w')
        io.write_snapshot(self.particles)











    def show(self, nbins=32):
        import numpy as np
        from scipy import optimize
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        mass = self.imf._mtot * self.particles["body"].mass.copy()

        ###################################

        (hist, bins) = np.histogram(np.log10(mass), bins=nbins)
        linbins = np.power(10.0, bins)
        where_gt_zero = np.where(hist > 0)

        fitfunc = lambda k, m: k * self.imf.func(m)
        errfunc = lambda k, m, y: fitfunc(k, m)[where_gt_zero] - y[where_gt_zero]
        k0 = 1.0
        k1, success = optimize.leastsq(errfunc, k0, args=(linbins[:-1], hist))
        x = np.logspace(np.log10(self.imf.min_mlow),
                        np.log10(self.imf.max_mhigh),
                        num=128, base=10.0)
        y = fitfunc(k1, x)

        ###################################
        # IMF plot

        fig = plt.figure(figsize=(13.5, 6))
        ax1 = fig.add_subplot(1,2,1)
        ax1.plot(bins[where_gt_zero], np.log10(hist[where_gt_zero]),
                 'bo', label='IMF sample')
        ax1.plot(np.log10(x), np.log10(y), 'r--',
                 label='IMF distribution', linewidth=1.5)
        ax1.grid(True)
        ax1.set_xlabel(r'$\log_{10}(m)$', fontsize=18)
        ax1.set_ylabel(r'$\log_{10}(dN/d\log_{10}(m))$', fontsize=18)
        ax1.legend(loc='lower left', shadow=True,
                   fancybox=True, borderaxespad=0.75)

        ###################################

        b = self.particles["body"]
        n = len(b)
        x = b.pos[:,0]
        y = b.pos[:,1]
        radius = 2 * n * b.mass
        color = n * b.mass

        ###################################
        # Scatter plot

        ax2 = fig.add_subplot(1,2,2)
#        ax.set_axis_bgcolor('0.75')
        ax2.scatter(x, y, c=color, s=radius, cmap='gist_rainbow',
                    alpha=0.75, label=r'$Stars$')
        circle = Circle((0, 0), 1, facecolor='none',
                        edgecolor=(1,0.25,0), linewidth=1.5, label=r'$R_{Vir}$')
        ax2.add_patch(circle)

        ax2.set_xlim(-4, +4)
        ax2.set_ylim(-4, +4)

        ax2.set_xlabel(r'$x$', fontsize=18)
        ax2.set_ylabel(r'$y$', fontsize=18)
        ax2.legend(loc='upper right', shadow=True,
                   fancybox=True, borderaxespad=0.75)

        ###################################
        # Show
        plt.savefig('show.png', bbox_inches="tight")
#        plt.savefig('show.pdf', bbox_inches="tight")
        plt.show()
        plt.close()



########## end of file ##########
