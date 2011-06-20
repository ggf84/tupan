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


__all__ = ['Plummer']


def scale_mass(bodies, m_scale):
    bodies.mass *= m_scale

def scale_pos(bodies, r_scale):
    bodies.pos *= r_scale

def scale_vel(bodies, v_scale):
    bodies.vel *= v_scale

def scale_to_virial(bodies, ekin, epot, etot):
    scale_vel(bodies, math.sqrt(-0.5*epot/ekin))
    ekin = -0.5*epot
    etot = ekin + epot
    scale_pos(bodies, etot/(-0.25))
    bodies.set_phi(bodies)
    epot = bodies.get_total_epot()
    scale_vel(bodies, math.sqrt(0.5*abs(epot/etot)))


def scale_to_nbody_units(bodies):
#    scale_mass(bodies, 1.0/bodies.get_mass())
#    bodies.set_phi(bodies)

    e = bodies.get_total_energies()
    print(e.kin, e.pot, e.tot)
    scale_to_virial(bodies, e.kin, e.pot, e.tot)
    e = bodies.get_total_energies()
    print(e.kin, e.pot, e.tot)





class Plummer(object):
    """  """

    def __init__(self, num, imf, mfrac=0.999, epsf=0.0, seed=None):
        from pynbody.particles import Body
        self.num = num
        self.imf = imf
        self.mfrac = mfrac
        self.epsf = epsf
        self._body = Body(num)
        random.seed(seed)


    def set_eps2(self, mass):
        eps = self.epsf * mass
        return 0.5 * (eps**2)


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
        self._body.index[:] = ilist

        s = random.getstate()

        # set mass
        self._body.mass[:] = self.imf.sample(n)
        self._body.mass /= self._body.get_total_mass()
        # set eps2
        self._body.eps2[:] = self.set_eps2(self._body.mass.copy())

        random.setstate(s)

        # set pos
        self._body.pos[:] = [self.set_pos(i) for i in random.sample(ilist,n)]

#        mcum = self._body.mass.cumsum()
#        mcum /= mcum.max()
#        mcum *= self.mfrac
#        self._body.pos[:] = [self.set_pos2(i) for i in random.sample(mcum, n)]

        # set phi
        self._body.set_phi(self._body)
        # set vel
        self._body.vel[:] = [self.set_vel(p) for p in self._body.phi]


    def make_plummer(self):
        self.set_bodies()
        self._body.reset_center_of_mass()
        scale_to_nbody_units(self._body)
        self._body.set_acc(self._body)


    def write_snapshot(self, fname='plummer.hdf5'):
        from pynbody.particles import Particles
        data = Particles()
        data.set_members(self._body)
        io = HDF5IO(fname, 'w')
        io.write_snapshot(data)











    def show(self, nbins=32):
        import numpy as np
        from scipy import optimize
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        mass = self.imf._mtot * self._body.mass.copy()

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

        fig = plt.figure(figsize=(14, 6.25))
        # semilogx
        ax1 = fig.add_subplot(2,2,1)
        ax1.semilogx(linbins[:-1], hist, 'bo', label='IMF sample')
        ax1.semilogx(x, y, 'r--', linewidth=1, label='IMF distribution')
        ax1.grid(True)
        ax1.set_ylabel(r'$dN/d\log(m)$')
        ax1.legend(loc='upper right', borderaxespad=1.5, shadow=True, fancybox=True)
        # loglog
        ax2 = fig.add_subplot(2,2,3, sharex=ax1)
        ax2.loglog(linbins[:-1], hist, 'bo', label='IMF sample')
        ax2.loglog(x, y, 'r--', linewidth=1, label='IMF distribution')
        ax2.grid(True)
        ax2.set_xlabel(r'$\log(m)$')
        ax2.set_ylabel(r'$\log(dN/d\log(m))$')
        ax2.text(0.25, 0.2,
                 '{0} bins\n{1} stars'.format(nbins, self.num),
                 fontsize=14,
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=ax2.transAxes)

        fig.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

        ###################################

        b = self._body
        n = len(b)
        x = b.pos[:,0]
        y = b.pos[:,1]
        radius = 0.5 * n * np.sqrt(b.eps2)
        color = n * b.mass

        ###################################
        # Scatter plot

        ax = fig.add_subplot(1,2,2)
#        ax.set_axis_bgcolor('0.75')
        ax.scatter(x, y, c=color, s=radius, cmap='gist_rainbow',
                   alpha=0.75, label=r'$Stars$')
        circle = Circle((0, 0), 1, facecolor='none',
                        edgecolor=(1,0.25,0), linewidth=1, label=r'$R_{Vir}$')
        ax.add_patch(circle)

        ax.set_xlim(-4, +4)
        ax.set_ylim(-4, +4)

        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.legend(loc='upper right', borderaxespad=1.5, shadow=True, fancybox=True)

        fig.subplots_adjust(left=0.08, right=0.95, bottom=0.11, top=0.95)

        ###################################
        # Show
        plt.savefig('show.png', bbox_inches="tight")
        plt.show()
        plt.close()



########## end of file ##########
