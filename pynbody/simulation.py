#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TODO.
"""

from __future__ import print_function
import sys
import math
import gzip
import pickle
from ggf84decor import selftimer
from pynbody.io import HDF5IO
from pynbody.integrator import (METH_NAMES, METHS)


__all__ = ['Simulation']


def myprint(data, fname, fmode):
    if fname == '<stdout>':
        print(data, file=sys.stdout)
    elif fname == '<stderr>':
        print(data, file=sys.stderr)
    else:
        with open(fname, fmode) as fobj:
            print(data, file=fobj)




class Diagnostic(object):
    """

    """
    def __init__(self, fname, particles):
        self.fname = fname

        particles.set_phi(particles)
        self.e0 = particles.get_total_energies()
        self.rcom0 = particles.get_center_of_mass_pos()
        self.vcom0 = particles.get_center_of_mass_vel()
        self.lmom0 = particles.get_total_linmom()
        self.amom0 = particles.get_total_angmom()

        self.ceerr = 0.0
        self.count = 0

        self.print_header()


    def __repr__(self):
        return '{0}'.format(self.__dict__)


    def print_header(self):
        fmt = '{0:10s} '\
              '{1:10s} {2:10s} {3:13s} '\
              '{4:10s} {5:10s} {6:10s} '\
              '{7:9s} {8:9s} {9:9s} '\
              '{10:9s} {11:9s} {12:9s} '\
              '{13:9s} {14:9s} {15:9s} '\
              '{16:9s} {17:9s} {18:9s}'
        myprint(fmt.format('#00:time',
                           '#01:ekin', '#02:epot', '#03:etot',
                           '#04:evir', '#05:eerr', '#06:geerr',
                           '#07:rcomX', '#08:rcomY', '#09:rcomZ',
                           '#10:vcomX', '#11:vcomY', '#12:vcomZ',
                           '#13:lmomX', '#14:lmomY', '#15:lmomZ',
                           '#16:amomX', '#17:amomY', '#18:amomZ'),
                self.fname, 'w')


    def print_diagnostic(self, time, particles):
        particles.set_phi(particles)
        e = particles.get_total_energies()
        rcom = particles.get_center_of_mass_pos()
        vcom = particles.get_center_of_mass_vel()
        lmom = particles.get_total_linmom()
        amom = particles.get_total_angmom()

        ejump = particles.get_total_energy_jump()
        eerr = ((e.tot-self.e0.tot) + ejump)/(-e.pot)
        self.ceerr += eerr**2
        self.count += 1
        geerr = math.sqrt(self.ceerr / self.count)
        dRcom = (rcom-self.rcom0) + particles.get_com_pos_jump()
        dVcom = (vcom-self.vcom0) + particles.get_com_vel_jump()
        dLmom = (lmom-self.lmom0) + particles.get_total_linmom_jump()
        dAmom = (amom-self.amom0) + particles.get_total_angmom_jump()

        fmt = '{time:< 10.3e} '\
              '{ekin:< 10.3e} {epot:< 10.3e} {etot:< 13.6e} '\
              '{evir:< 10.3e} {eerr:< 10.3e} {geerr:< 10.3e} '\
              '{rcom[0]:< 9.2e} {rcom[1]:< 9.2e} {rcom[2]:< 9.2e} '\
              '{vcom[0]:< 9.2e} {vcom[1]:< 9.2e} {vcom[2]:< 9.2e} '\
              '{lmom[0]:< 9.2e} {lmom[1]:< 9.2e} {lmom[2]:< 9.2e} '\
              '{amom[0]:< 9.2e} {amom[1]:< 9.2e} {amom[2]:< 9.2e}'
        myprint(fmt.format(time=time,
                           ekin=e.kin+ejump, epot=e.pot,
                           etot=e.tot+ejump, evir=e.vir+ejump,
                           eerr=eerr, geerr=geerr, rcom=dRcom,
                           vcom=dVcom, lmom=dLmom, amom=dAmom),
                self.fname, 'a')



class Simulation(object):
    """
    The Simulation class is the top level class for N-body simulations.
    """
    def __init__(self, args, viewer):
        self.args = args
        self.viewer = viewer

        # Read the initial conditions.
        io = HDF5IO(self.args.input)
        particles = io.read_snapshot()

        # Set the method of integration.
        self.Integrator = METHS[METH_NAMES.index(self.args.meth)]

        # Initializes the integrator.
        self.integrator = self.Integrator(self.args.eta, 0.0, particles)

        # Initializes the diagnostic of the simulation.
        self.dia = Diagnostic(self.args.log_file, particles)
        self.dia.print_diagnostic(self.integrator.time, particles)

        # Initializes snapshots output.
        self.iosnaps = HDF5IO('snapshots.hdf5')
        self.iosnaps.snap_number = 0
        self.iosnaps.write_snapshot(particles)

        # Initializes times for output a couple of things.
        self.dt_gl = 1.0 / self.args.gl_freq
        self.dt_dia = 1.0 / self.args.diag_freq
        self.dt_res = 1.0 / self.args.res_freq
        self.oldtime_gl = self.integrator.time
        self.oldtime_dia = self.integrator.time
        self.oldtime_res = self.integrator.time


    def __getstate__(self):
        # This is just a method of pickle-related behavior.
        sdict = self.__dict__.copy()
        return sdict


    def __setstate__(self, sdict):
        # This is just a method of pickle-related behavior.
        integrator = sdict["integrator"]
        integrator.particles = integrator.gather().copy()
        self.__dict__.update(sdict)
        self.integrator = integrator


    @selftimer
    def evolve(self):
        """

        """
        if self.viewer:
            self.viewer.initialize()

        while (self.integrator.time < self.args.tmax):
            self.integrator.step()
            if (self.integrator.time - self.oldtime_gl >= self.dt_gl):
                self.oldtime_gl += self.dt_gl
                if self.viewer:
                    self.viewer.show_event(self.integrator)
            if (self.integrator.time - self.oldtime_dia >= self.dt_dia):
                self.oldtime_dia += self.dt_dia
                particles = self.integrator.gather()
                self.dia.print_diagnostic(self.integrator.time, particles)
                self.iosnaps.snap_number += 1
                self.iosnaps.write_snapshot(particles)
            if (self.integrator.time - self.oldtime_res >= self.dt_res):
                self.oldtime_res += self.dt_res
#                with gzip.open('restart.pkl.gz', 'wb') as fobj:
#                    pickle.dump(self, fobj, protocol=pickle.HIGHEST_PROTOCOL)

                # <py2.6>
                fobj = gzip.open('restart.pkl.gz', 'wb')
                pickle.dump(self, fobj, protocol=pickle.HIGHEST_PROTOCOL)
                fobj.close()
                # </py2.6>

        if self.viewer:
            self.viewer.enter_main_loop()


########## end of file ##########
