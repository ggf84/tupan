# -*- coding: utf-8 -*-
#

"""
TODO.
"""

import logging
import numpy as np
from ..units import ureg
from ..particles.system import ParticleSystem


LOGGER = logging.getLogger(__name__)


class Plummer(object):
    """  """

    def __init__(self, eps, imf, seed=None, mfrac=0.999, softening_type=0):
        self.eps2 = eps*eps
        self.imf = imf
        self.mfrac = mfrac
        self.softening_type = softening_type
        if seed:
            np.random.seed(seed)

    def set_eps2(self, mass):
        n = len(mass)
        if self.softening_type == 0:
            # eps2 ~ cte
            eps2 = np.ones(n)
        elif self.softening_type == 1:
            # eps2 ~ m^2 ~ 1/n^2 if m ~ 1/n
            eps2 = mass**2
        elif self.softening_type == 2:
            # eps2 ~ m/n ~ 1/n^2 if m ~ 1/n
            eps2 = mass / n
        elif self.softening_type == 3:
            # eps2 ~ (m/n^2)^(2/3) ~ 1/n^2 if m ~ 1/n
            eps2 = (mass / n**2)**(2.0/3)
        elif self.softening_type == 4:
            # eps2 ~ (1/(m*n^2))^2 ~ 1/n^2 if m ~ 1/n
            eps2 = (1.0 / (mass * n**2))**2
        else:
            LOGGER.critical(
                "Unexpected value for softening_type: %d.",
                self.softening_type)
            raise ValueError(
                "Unexpected value for softening_type: {}.".format(
                    self.softening_type))

        # normalizes by the provided scale of eps2
        eps2 *= self.eps2 / np.mean(eps2)

        # return half of real value in order to avoid to do this in force loop.
        return eps2 / 2

    def set_pos(self, irand):
        n = len(irand)
        mfrac = self.mfrac
        mrand = (irand + np.random.random(n)) * mfrac / n
        radius = 1.0 / np.sqrt(np.power(mrand, -2.0 / 3.0) - 1.0)
        theta = np.arccos(np.random.uniform(-1.0, 1.0, size=n))
        phi = np.random.uniform(0.0, 2.0 * np.pi, size=n)
        rx = radius * np.sin(theta) * np.cos(phi)
        ry = radius * np.sin(theta) * np.sin(phi)
        rz = radius * np.cos(theta)
        return (rx, ry, rz)

    def set_vel(self, pot):
        count = 0
        n = len(pot)
        rnd = np.empty(n)
        while count < n:
            r1 = np.random.random()
            r2 = np.random.random()
            if r2 < r1:
                rnd[count] = r2
                count += 1
        velocity = np.sqrt(-2 * rnd * pot)
        theta = np.arccos(np.random.uniform(-1.0, 1.0, size=n))
        phi = np.random.uniform(0.0, 2.0 * np.pi, size=n)
        vx = velocity * np.sin(theta) * np.cos(phi)
        vy = velocity * np.sin(theta) * np.sin(phi)
        vz = velocity * np.cos(theta)
        return (vx, vy, vz)

    def make_model(self, n):
        """  """
        ilist = np.arange(n)

        ps = ParticleSystem(n)
        b = ps.bodies

        srand = np.random.get_state()

        # set mass
        b.mass[...] = self.imf.sample(n) * ureg.M_sun

        # set eps2
        b.eps2[...] = self.set_eps2(b.mass) * ureg.pc**2

        np.random.set_state(srand)

        # set pos
        pos = self.set_pos(np.random.permutation(ilist))
        b.pos[...] = pos * ureg.pc

        # set phi
        ps.set_phi(ps)

        # set vel
        vel = self.set_vel(b.phi.m_as('(uL/uT)**2'))
        b.vel[...] = vel * ureg('uL/uT')

        ps.dynrescale_virial_radius(1.0 * ureg.uL)

        ps.com_to_origin()
        ps.scale_to_virial()

#        print(ps.kinetic_energy)
#        print(ps.potential_energy)
#        print(ps.virial_radius.to('pc'))
#        print(np.sum(b.mass).to('M_sun'))
#        print(np.mean(b.mass).to('M_sun'))
#        print(np.std(b.vel[0]).to('km/s'))
#        print(np.std(b.vel[1]).to('km/s'))
#        print(np.std(b.vel[2]).to('km/s'))

        return ps

    def show(self, bodies, nbins=32):
        from scipy import optimize
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        r = bodies.pos.m_as('pc')
        mass = bodies.mass.m_as('M_sun')

        ###################################

        hist, bins = np.histogram(np.log10(mass), bins=nbins)
        linbins = np.power(10.0, bins)
        selection = np.where(hist > 0)

        def fitfunc(k, m):
            return k * self.imf.func(m)

        def errfunc(k, m, y):
            return fitfunc(k, m)[selection] - y[selection]

        k0 = 1.0
        k1, _ = optimize.leastsq(errfunc, k0, args=(linbins[:-1], hist))
        x = np.logspace(np.log10(self.imf.min_mlow),
                        np.log10(self.imf.max_mhigh),
                        num=128, base=10.0)
        y = fitfunc(k1, x)

        ###################################
        # IMF plot

        fig = plt.figure(figsize=(8, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(bins[selection], np.log10(hist[selection]),
                 'bo', label='IMF sample')
        ax1.plot(np.log10(x), np.log10(y), 'r--',
                 label='IMF distribution', linewidth=1.5)
        ax1.grid(True)
        ax1.set_xlabel(r'$\log_{10}(m)$', fontsize=18)
        ax1.set_ylabel(r'$\log_{10}(dN/d\log_{10}(m))$', fontsize=18)
        ax1.legend(loc='lower left', shadow=True,
                   fancybox=True, borderaxespad=0.75)

        ###################################

        radius = 2 * mass
        color = mass

        ###################################
        # Scatter plot

        ax2 = fig.add_subplot(1, 2, 2)
#        ax.set_axis_bgcolor('0.75')
        ax2.scatter(r[0], r[1], c=color, s=radius, cmap='gist_rainbow',
                    alpha=0.5, label=r'$Stars$')
        circle = Circle(
            (0, 0), 1,
            facecolor='none',
            edgecolor=(1, 0.25, 0),
            linewidth=1.5,
            label=r'$R_{Vir}$'
        )
        ax2.add_patch(circle)

        ax2.set_xlim(-4, +4)
        ax2.set_ylim(-4, +4)

        ax2.set_xlabel(r'$x$', fontsize=18)
        ax2.set_ylabel(r'$y$', fontsize=18)
        ax2.legend(loc='upper right', shadow=True,
                   fancybox=True, borderaxespad=0.75)

        plt.tight_layout()
        ###################################
        # Show
        plt.savefig('show.png', bbox_inches="tight")
#        plt.savefig('show.pdf', bbox_inches="tight")
        plt.show()
        plt.close()


def make_plummer(n, eps, imf,
                 seed=None, mfrac=0.999,
                 softening_type=0, show=False):
    if n < 2:
        n = 2
    from tupan.ics.imf import IMF
    imf = getattr(IMF, imf[0])(*imf[1:])
    p = Plummer(eps, imf, seed=seed, mfrac=mfrac,
                softening_type=softening_type)
    ps = p.make_model(n)
    if show:
        p.show(ps.bodies)
    return ps


# -- End of File --
