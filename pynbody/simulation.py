#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The Simulation class"""

from __future__ import print_function
import sys
import pickle
from ggf84decor import selftimer
from pynbody.io import HDF5IO
from pynbody.integrator import (METH_NAMES, METHS)


RUN_MODES = ['newrun', 'restart']


class Diagnostic(object):
    """

    """
    def __init__(self, fobj, e0, rcom0, lmom0, amom0):
        self.fobj = fobj
        self.e0 = e0
        self.rcom0 = rcom0
        self.lmom0 = lmom0
        self.amom0 = amom0


    def __repr__(self):
        return '{0}'.format(self.__dict__)


    def print_header(self):
        fmt = '{0:16s} '\
              '{1:16s} {2:12s} {3:12s} {4:11s} '\
              '{5:10s} {6:10s} {7:10s} '\
              '{8:10s} {9:10s} {10:10s} '\
              '{11:10s} {12:10s} {13:10s}'
        print(fmt.format('#time:00',
                         '#etot:01', '#ekin:02', '#epot:03', '#eerr:04',
                         '#rcomX:05', '#rcomY:06', '#rcomZ:07',
                         '#lmomX:08', '#lmomY:09', '#lmomZ:10',
                         '#amomX:11', '#amomY:12', '#amomZ:13'),
              file=self.fobj)


    def print_diagnostic(self, time, particles):
        particles['body'].set_phi(particles['body'])
        e = particles['body'].get_energies()
        e1 = e.tot
        rcom1 = particles['body'].get_Rcenter_of_mass()
        lmom1 = particles['body'].get_linear_mom()
        amom1 = particles['body'].get_angular_mom()

        eerr = (e1-self.e0)/abs(self.e0)
        rcom = (rcom1-self.rcom0)
        lmom = (lmom1-self.lmom0)
        amom = (amom1-self.amom0)

        fmt = '{time:< 16.10g} '\
              '{etot:< 16.10g} {ekin:< 12.6g} {epot:< 12.6g} {eerr:< 11.5g} '\
              '{rcom[0]:< 10.4g} {rcom[1]:< 10.4g} {rcom[2]:< 10.4g} '\
              '{lmom[0]:< 10.4g} {lmom[1]:< 10.4g} {lmom[2]:< 10.4g} '\
              '{amom[0]:< 10.4g} {amom[1]:< 10.4g} {amom[2]:< 10.4g}'
        print(fmt.format(time=time,
                         etot=e.tot, ekin=e.kin, epot=e.pot, eerr=eerr,
                         rcom=rcom, lmom=lmom, amom=amom),
              file=self.fobj)



class Simulation(object):
    """The Simulation class is the top level class for N-body simulations"""

    def __init__(self, args):
        self.args = args

        # Set the method of integration.
        if self.args.meth in METH_NAMES:
            self.Integrator = METHS[METH_NAMES.index(self.args.meth)]
        else:
            print('Typo or invalid name of the method of integration.')
            print('Available methods:', METH_NAMES)
            print('exiting...')
            sys.exit(1)


        io = HDF5IO(self.args.input)
        particles = io.read_snapshot()

        if self.args.smod == 'restart':
            print('#'*25)
            print(self.args.__dict__)
            print('#'*25)

            self.snapcount = 16
            e = particles['body'].get_energies()
            e0 = e.tot
            rcom0 = particles['body'].get_Rcenter_of_mass()
            lmom0 = particles['body'].get_linear_mom()
            amom0 = particles['body'].get_angular_mom()
            self.dia = Diagnostic(self.args.log, e0, rcom0, lmom0, amom0)
        else:
            print('#'*25)
            print(self.args.__dict__)
            print('#'*25)

            self.snapcount = 0
            e = particles['body'].get_energies()
            e0 = e.tot
            rcom0 = particles['body'].get_Rcenter_of_mass()
            lmom0 = particles['body'].get_linear_mom()
            amom0 = particles['body'].get_angular_mom()
            self.dia = Diagnostic(self.args.log, e0, rcom0, lmom0, amom0)
            self.dia.print_header()
            self.dia.print_diagnostic(0.0, particles)
            particles['body'].set_acc(particles['body'])
            particles['body'].stepdens[:,1] = particles['body'].stepdens[:,0].copy()
            io = HDF5IO('snapshots.hdf5')
            io.write_snapshot(particles, group_id=self.snapcount)

        iorestart = HDF5IO('restart.hdf5', 'w')
        iorestart.write_snapshot(particles)

        self.integrator = self.Integrator(0.0, self.args.eta, particles)


    @selftimer
    def evolve(self):
        """

        """
        diadt = 1.0 / self.args.dia
        io = HDF5IO('snapshots.hdf5')
        iorestart = HDF5IO('restart.hdf5', 'w')

        while (self.integrator.time < self.args.tmax):
            old_restime = self.integrator.time
            while ((self.integrator.time - old_restime < self.args.resdt) and
                   (self.integrator.time < self.args.tmax)):
                self.snapcount += 1
                old_diatime = self.integrator.time
                while ((self.integrator.time - old_diatime < diadt) and
                       (self.integrator.time < self.args.tmax)):
                    self.integrator.step()
                particles = self.integrator.gather()
                self.dia.print_diagnostic(self.integrator.time, particles)
                io.write_snapshot(particles, group_id=self.snapcount)
            if (self.integrator.time - old_restime >= self.args.resdt):
                iorestart.write_snapshot(particles)


    def run(self):
        """Initialize a N-body simulation"""
        print('running...')

        io = HDF5IO('input.hdf5')
        myuniverse = io.read_snapshot()

        io = HDF5IO('output.hdf5')
        io.write_snapshot(myuniverse)
        mynewuniverse = io.read_snapshot()

        for i in zip(myuniverse['blackhole'], mynewuniverse['blackhole']):
            for j in zip(i[0], i[1]):
                print(j[0] == j[1])

        print('running... done')


########## end of file ##########
