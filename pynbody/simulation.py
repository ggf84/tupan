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
from pprint import pprint
from .io.hdf5io import HDF5IO
from .integrator import Integrator
from .lib.utils.timing import timings


__all__ = ['Simulation']


@timings
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
        fmt = '{0:10s} {1:10s} '\
              '{2:10s} {3:10s} {4:13s} '\
              '{5:10s} {6:10s} {7:10s} '\
              '{8:9s} {9:9s} {10:9s} '\
              '{11:9s} {12:9s} {13:9s} '\
              '{14:9s} {15:9s} {16:9s} '\
              '{17:9s} {18:9s} {19:9s}'
        myprint(fmt.format('#00:time', '#01:tstep',
                           '#02:ekin', '#03:epot', '#04:etot',
                           '#05:evir', '#06:eerr', '#07:geerr',
                           '#08:rcomX', '#09:rcomY', '#10:rcomZ',
                           '#11:vcomX', '#12:vcomY', '#13:vcomZ',
                           '#14:lmomX', '#15:lmomY', '#16:lmomZ',
                           '#17:amomX', '#18:amomY', '#19:amomZ'),
                self.fname, 'w')

    @timings
    def print_diagnostic(self, time, tstep, particles):
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

        fmt = '{time:< 10.3e} {tstep:< 10.3e} '\
              '{ekin:< 10.3e} {epot:< 10.3e} {etot:< 13.6e} '\
              '{evir:< 10.3e} {eerr:< 10.3e} {geerr:< 10.3e} '\
              '{rcom[0]:< 9.2e} {rcom[1]:< 9.2e} {rcom[2]:< 9.2e} '\
              '{vcom[0]:< 9.2e} {vcom[1]:< 9.2e} {vcom[2]:< 9.2e} '\
              '{lmom[0]:< 9.2e} {lmom[1]:< 9.2e} {lmom[2]:< 9.2e} '\
              '{amom[0]:< 9.2e} {amom[1]:< 9.2e} {amom[2]:< 9.2e}'
        myprint(fmt.format(time=time, tstep=tstep,
                           ekin=e.kin+ejump, epot=e.pot,
                           etot=e.tot+ejump, evir=e.vir+ejump,
                           eerr=eerr, geerr=geerr, rcom=dRcom,
                           vcom=dVcom, lmom=dLmom, amom=dAmom),
                self.fname, 'a')



class Simulation(object):
    """
    The Simulation class is the top level class for N-body simulations.
    """
    @timings
    def __init__(self, args, viewer):
        self.args = args
        self.viewer = viewer

        print('#'*40, file=sys.stderr)
        pprint(args.__dict__, stream=sys.stderr)
        print('#'*40, file=sys.stderr)

        # Read the initial conditions.
        ic = HDF5IO(self.args.input)
        particles = ic.read_snapshot()

        # Initializes the integrator.
        self.integrator = Integrator(self.args.eta, 0.0, particles,
                                     meth=self.args.meth)

        # Initializes the diagnostic of the simulation.
        self.dia = Diagnostic(self.args.log_file, particles)
        self.dia.print_diagnostic(self.integrator.time,
                                  self.integrator.tstep,
                                  particles)

        # Initializes snapshots output.
        self.io = HDF5IO("snapshots")
        self.snap_count = 0
        snap_name = "snap_" + str(self.snap_count).zfill(5)
        self.io.write_snapshot(particles, snap_name, self.integrator.time)

        # Initializes times for output a couple of things.
        self.dt_gl = 1.0 / self.args.gl_freq
        self.dt_dia = 1.0 / self.args.diag_freq
        self.dt_res = 1.0 / self.args.res_freq
        self.oldtime_gl = self.integrator.time
        self.oldtime_dia = self.integrator.time
        self.oldtime_res = self.integrator.time


    @timings
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
                    particles = self.integrator.particles
                    self.viewer.show_event(particles.copy())
            if (self.integrator.time - self.oldtime_dia >= self.dt_dia):
                self.oldtime_dia += self.dt_dia
                particles = self.integrator.particles
                self.dia.print_diagnostic(self.integrator.time,
                                          self.integrator.tstep,
                                          particles)
                self.snap_count += 1
                snap_name = "snap_" + str(self.snap_count).zfill(5)
                self.io.write_snapshot(particles, snap_name,
                                       self.integrator.time)
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


    def print_timings(self):
        print(timings, file=sys.stderr)


########## end of file ##########
