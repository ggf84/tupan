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

    def to_xyz(self, r):
        n = len(r)
        theta = np.arccos(np.random.uniform(-1.0, 1.0, size=n))
        phi = np.random.uniform(0.0, 2.0 * np.pi, size=n)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.array([x, y, z])

    def get_coordinates(self, n):
        # set radius
        i = np.arange(n)
        m_min = (i + 0) / n
        m_max = (i + 1) / n
        mfrac = self.mfrac
        mfrac *= np.random.uniform(m_min, m_max, size=n)
        radius = 1.0 / np.sqrt(pow(mfrac, -2.0/3.0) - 1.0)

        # set velocity
        q = np.zeros(0)
        nq = len(q)
        while nq < n:
            x = np.random.uniform(0.0, 1.0, size=n-nq)
            y = np.random.uniform(0.0, 0.1, size=n-nq)
            g = x * x * pow(1.0 - x * x, 3.5)
            q = np.concatenate([q, x.compress(y < g)])
            nq = len(q)
        velocity = q * np.sqrt(2.0) * pow(1 + radius * radius, -0.25)

        # xyz components
        return self.to_xyz(radius), self.to_xyz(velocity)

    def make_model(self, n):
        """

        """
        ps = ParticleSystem(n)
        b = ps.bodies

        # set mass
        b.mass[...] = self.imf

        # set eps2
        b.eps2[...] = self.set_eps2(b.mass) * ureg('uL**2')

        # set pos, vel
        pos, vel = self.get_coordinates(n)
        b.pos[...] = pos * ureg('uL')
        b.vel[...] = vel * ureg('uL/uT')

        # rescaling virial radius
        scale_factor = 16 / (3 * np.pi)
        b.pos[...] /= scale_factor
        b.vel[...] *= np.sqrt(scale_factor)

        ps.com_to_origin()
        ps.scale_to_standard()

        ps.set_phi(ps)
        print(ps.kinetic_energy)
        print(ps.potential_energy)
        print(ps.virial_radius.to('pc'))
        print(np.min(b.mass).to('M_sun'))
        print(np.median(b.mass.m_as('M_sun')) * ureg('M_sun'))
        print(np.mean(b.mass).to('M_sun'))
        print(np.std(b.mass).to('M_sun'))
        print(np.max(b.mass).to('M_sun'))
        print(np.sum(b.mass).to('M_sun'))
        print(np.sum(b.mass).to('uM'))
        print(np.std(b.vel[0]).to('km/s'))
        print(np.std(b.vel[1]).to('km/s'))
        print(np.std(b.vel[2]).to('km/s'))

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
    p = Plummer(eps, imf, seed=seed, mfrac=mfrac,
                softening_type=softening_type)
    ps = p.make_model(n)
    if show:
        p.show(ps.bodies)
    return ps


# -- End of File --
