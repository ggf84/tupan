# -*- coding: utf-8 -*-
#

"""
TODO.
"""


from __future__ import print_function
import yaml
from ..lib.utils.timing import decallmethods, timings


__all__ = ['PSDFIO']


@decallmethods(timings)
class PSDFIO(object):
    """

    """
    def __init__(self, fname):
        self.fname = fname

    def dump(self, ps, fmode='a'):
        """
        Serialize particle objects into a YAML stream.
        """
        data = Stream.to_dumper(ps)
        with open(self.fname, fmode) as fobj:
            yaml.dump_all(data,
                          stream=fobj,
                          default_flow_style=False,
                          explicit_start=True)

    def load(self):
        """
        Load a YAML stream into particle objects.
        """
        with open(self.fname, 'r') as fobj:
            data = [i for i in yaml.load_all(fobj)]
        ps = Stream.from_loader(data)
        return ps

    def to_hdf5(self):
        """
        Converts a YAML stream into a HDF5 one.
        """
        from . import hdf5io
        fname = self.fname.replace('.psdf', '.hdf5')
        stream = hdf5io.HDF5IO(fname, fmode='w')
        try:
            ps = self.load_snapshot()
            stream.dump_snapshot(ps)
        except:
            ps = self.load_worldline()
            stream.dump_worldline(ps)


#@decallmethods(timings)
class Stream(yaml.YAMLObject):
    """

    """
    yaml_tag = '!Particle'

    @classmethod
    def to_yaml(cls, dumper, data):
        """
        Convert a Python object to a representation node.
        """
        # default attributes
        attributes = {'id': data.id, 'm': data.m, 't': data.t_curr,
                      'r': data.r, 'v': data.v, 'a': data.a}
        # tupan's specific attributes
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

    @classmethod
    def to_dumper(cls, ps):
        data = []
        for (key, objs) in ps.items:
            if objs.n:
                for obj in objs:
                    od = cls()
                    attributes = obj.data.dtype.names
                    # default attributes
                    if 'id' in attributes:
                        od.id = int(obj.id)
                    if 'mass' in attributes:
                        od.m = float(obj.mass)
                    if 'pos' in attributes:
                        od.r = [float(obj.pos[0]),
                                float(obj.pos[1]),
                                float(obj.pos[2])]
                    if 'vel' in attributes:
                        od.v = [float(obj.vel[0]),
                                float(obj.vel[1]),
                                float(obj.vel[2])]
                    if 'acc' in attributes:
                        od.a = [float(obj.acc[0]),
                                float(obj.acc[1]),
                                float(obj.acc[2])]
                    if 't_curr' in attributes:
                        od.t_curr = float(obj.t_curr)
                    # tupan's specific attributes
                    od.type = key
                    if 'dt_prev' in attributes:
                        od.dt_prev = float(obj.dt_prev)
                    if 'dt_next' in attributes:
                        od.dt_next = float(obj.dt_next)
                    if 'eps2' in attributes:
                        od.eps2 = float(obj.eps2)
                    if 'phi' in attributes:
                        od.pot = float(obj.phi)
                    if 'spin' in attributes:
                        od.s = [float(obj.spin[0]),
                                float(obj.spin[1]),
                                float(obj.spin[2])]

                    data.append(od)
        return data

    @classmethod
    def from_yaml(cls, loader, node):
        """
        Convert a representation node to a Python object.
        """
        return loader.construct_mapping(node, deep=True)

    @classmethod
    def from_loader(cls, data):
        from tupan.particles.allparticles import System
        from tupan.particles.sph import Sphs
        from tupan.particles.star import Stars
        from tupan.particles.blackhole import Blackholes

        def set_attributes(obj, index, item):
            obj.id[index] = item['id']
            obj.mass[index] = item['m']
            obj.t_curr[index] = item['t']
            obj.pos[index] = item['r']
            obj.vel[index] = item['v']
#            obj.acc[index] = item['a']

            if 'a' in item:
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

        ps = System()

        for item in data:
            if 'type' in item:
                # set Body
                if item['type'] == 'body':
                    if ps['body']:
                        olen = len(ps['body'])
                        ps['body'].append(Stars(1))
                        set_attributes(ps['body'], olen, item)
                    else:
                        ps['body'] = Stars(1)
                        set_attributes(ps['body'], 0, item)
                # set BlackHole
                elif item['type'] == 'blackhole':
                    if ps['blackhole']:
                        olen = len(ps['blackhole'])
                        ps['blackhole'].append(Blackholes(1))
                        set_attributes(ps['blackhole'], olen, item)
                    else:
                        ps['blackhole'] = Blackholes(1)
                        set_attributes(ps['blackhole'], 0, item)
                # set Sph
                elif item['type'] == 'sph':
                    if ps['sph']:
                        olen = len(ps['sph'])
                        ps['sph'].append(Sphs(1))
                        set_attributes(ps['sph'], olen, item)
                    else:
                        ps['sph'] = Sphs(1)
                        set_attributes(ps['sph'], 0, item)
            else:
                print(
                    "Unspecified particle type! "
                    "Using by default the 'body' type."
                )
                if ps['body']:
                    olen = len(ps['body'])
                    ps['body'].append(Stars(1))
                    set_attributes(ps['body'], olen, item)
                else:
                    ps['body'] = Stars(1)
                    set_attributes(ps['body'], 0, item)

        return ps


########## end of file ##########
