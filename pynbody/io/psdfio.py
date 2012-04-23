#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

from __future__ import print_function
import yaml
from pynbody.lib.utils.timing import timings


__all__ = ['PSDFIO']


class PSDFIO(object):
    """

    """
    def __init__(self, fname):
        self.fname = fname


    @timings
    def dump(self, particles, fmode='a'):
        """
        Serialize particle objects into a YAML stream.
        """
        data = Stream.to_dumper(particles)
        with open(self.fname, fmode) as fobj:
            yaml.dump_all(data,
                          stream=fobj,
                          default_flow_style=False,
                          explicit_start=True)


    @timings
    def load(self):
        """
        Load a YAML stream into particle objects.
        """
        with open(self.fname, 'r') as fobj:
            data = [i for i in yaml.load_all(fobj)]
        particles = Stream.from_loader(data)
        return particles


    def to_hdf5(self):
        """
        Converts a YAML stream into a HDF5 one.
        """
        from . import hdf5io
        fname = self.fname.replace('.psdf', '.hdf5')
        stream = hdf5io.HDF5IO(fname)
        p = self.load()
        stream.dump(p, fmode='w')



class Stream(yaml.YAMLObject):
    """

    """
    yaml_tag = '!Particle'


    @classmethod
    @timings
    def to_yaml(cls, dumper, data):
        """
        Convert a Python object to a representation node.
        """
        # default attributes
        attributes = {'id': data.id, 'm': data.m, 't': data.t_curr,
                      'r': data.r, 'v': data.v, 'a': data.a}
        # PyNbody's specific attributes
        if hasattr(data, 'type'):
            attributes['type'] = data.type
        if hasattr(data, 'dt_prev'):
            attributes['dt_prev'] = data.dt_prev
        if hasattr(data, 'dt_next'):
            attributes['dt_next'] = data.dt_next
        if hasattr(data, 'eps2'):
            attributes['eps2'] = data.eps2
        if hasattr(data, 'pot'):
            attributes['pot'] = data.pot
        if hasattr(data, 's'):
            attributes['s'] = data.s
        # construct a representation node and return
        return dumper.represent_mapping(data.yaml_tag, attributes)

#        if data.type == 'body':
#            return dumper.represent_mapping(data.yaml_tag + '/Body', attributes)
#        elif data.type == 'blackhole':
#            return dumper.represent_mapping(data.yaml_tag + '/BlackHole', attributes)
#        elif data.type == 'sph':
#            return dumper.represent_mapping(data.yaml_tag + '/Sph', attributes)
#        else:
#            return dumper.represent_mapping(data.yaml_tag, attributes)


    @classmethod
    @timings
    def to_dumper(cls, particles):
        data = []
        for (key, objs) in particles.items():
            if objs:
                for obj in objs:
                    p = cls()
                    attributes = obj.data.dtype.names
                    # default attributes
                    if 'id' in attributes:
                        p.id = int(obj.id)
                    if 'mass' in attributes:
                        p.m = float(obj.mass)
                    if 'pos' in attributes:
                        p.r = [float(obj.pos[0]),
                               float(obj.pos[1]),
                               float(obj.pos[2])]
                    if 'vel' in attributes:
                        p.v = [float(obj.vel[0]),
                               float(obj.vel[1]),
                               float(obj.vel[2])]
                    if 'acc' in attributes:
                        p.a = [float(obj.acc[0]),
                               float(obj.acc[1]),
                               float(obj.acc[2])]
                    if 't_curr' in attributes:
                        p.t_curr = float(obj.t_curr)
                    # PyNbody's specific attributes
                    p.type = key
                    if 'dt_prev' in attributes:
                        p.dt_prev = float(obj.dt_prev)
                    if 'dt_next' in attributes:
                        p.dt_next = float(obj.dt_next)
                    if 'eps2' in attributes:
                        p.eps2 = float(obj.eps2)
                    if 'phi' in attributes:
                        p.pot = float(obj.phi)
                    if 'spin' in attributes:
                        p.s = [float(obj.spin[0]),
                               float(obj.spin[1]),
                               float(obj.spin[2])]

                    data.append(p)
        return data


    @classmethod
    @timings
    def from_yaml(cls, loader, node):
        """
        Convert a representation node to a Python object.
        """
        return loader.construct_mapping(node, deep=True)


    @classmethod
    @timings
    def from_loader(cls, data):
        from pynbody.particles import Particles
        from pynbody.particles.sph import Sph
        from pynbody.particles.body import Body
        from pynbody.particles.blackhole import BlackHole

        def set_attributes(obj, index, item):
            obj.id[index] = item['id']
            obj.mass[index] = item['m']
            obj.t_curr[index] = item['t']
            obj.pos[index] = item['r']
            obj.vel[index] = item['v']
            obj.acc[index] = item['a']

            if 'dt_prev' in item:
                obj.dt_prev[index] = item['dt_prev']
            if 'dt_next' in item:
                obj.dt_next[index] = item['dt_next']
            if 'eps2' in item:
                obj.eps2[index] = item['eps2']
            if 'pot' in item:
                obj.phi[index] = item['pot']
            if 's' in item:
                obj.spin[index] = item['s']

        p = Particles()

        for item in data:
            if 'type' in item:
                # set Body
                if item['type'] == 'body':
                    if p['body']:
                        olen = len(p['body'])
                        p['body'].append(Body(1))
                        set_attributes(p['body'], olen, item)
                    else:
                        p['body'] = Body(1)
                        set_attributes(p['body'], 0, item)
                # set BlackHole
                elif item['type'] == 'blackhole':
                    if p['blackhole']:
                        olen = len(p['blackhole'])
                        p['blackhole'].append(BlackHole(1))
                        set_attributes(p['blackhole'], olen, item)
                    else:
                        p['blackhole'] = BlackHole(1)
                        set_attributes(p['blackhole'], 0, item)
                # set Sph
                elif item['type'] == 'sph':
                    if p['sph']:
                        olen = len(p['sph'])
                        p['sph'].append(Sph(1))
                        set_attributes(p['sph'], olen, item)
                    else:
                        p['sph'] = Sph(1)
                        set_attributes(p['sph'], 0, item)
                else:
                    print("Unspecified particle type!")

        return p


########## end of file ##########
