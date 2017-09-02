# -*- coding: utf-8 -*-
#

"""
Utility functions for benchmarking extension kernels.
"""

import timeit
import numpy as np
from .particles.system import ParticleSystem
from .lib import extensions as ext


KERNEL = [
    ('Phi', {}),
    ('Acc', {'nforce': 1}),
    ('AccJrk', {'nforce': 2}),
    ('SnpCrk', {'nforce': 4}),
    ('Tstep', {'eta': 1/64}),
    ('PNAcc', {'pn': {'order': 7, 'clight': 128.0}}),
    ('Sakura', {'dt': 1/64, 'flag': -2}),
]


def set_particles(n):
    ps = ParticleSystem(n)
    b = ps.bodies
    b.mass[...] = np.random.random((n,))
    b.rdot[0][...] = np.random.random((3, n)) * 10
    b.rdot[1][...] = np.random.random((3, n)) * 10
    b.register_attribute('pnacc', '{nd}, {nb}', 'real_t')
    b.register_attribute('drdot', '2, {nd}, {nb}', 'real_t')
    return b


def benchmark(bench, n, backend, rect=False):
    np.random.seed(0)

    name, kwargs = bench

    name += '_rectangle' if rect else '_triangle'

    kernel = ext.make_extension(name, backend)

    ips = set_particles(n)
    jps = set_particles(n)

    print(f'\n# {name} (n = {n}):')
    elapsed = {'set': [], 'run': [], 'get': []}
    for i in range(3):
        if rect:
            t0 = timeit.default_timer()
            ibufs, (oargs, obufs) = kernel.set_args(ips, jps, **kwargs)
            t1 = timeit.default_timer()
            elapsed['set'].append(t1-t0)
        else:
            t0 = timeit.default_timer()
            ibufs, (oargs, obufs) = kernel.set_args(ips, **kwargs)
            t1 = timeit.default_timer()
            elapsed['set'].append(t1-t0)

        t0 = timeit.default_timer()
        kernel.run(ibufs+obufs)
        t1 = timeit.default_timer()
        elapsed['run'].append(t1-t0)

        t0 = timeit.default_timer()
        kernel.map_buffers(oargs, obufs)
        t1 = timeit.default_timer()
        elapsed['get'].append(t1-t0)

    for meth, values in elapsed.items():
        print(f"#\t{meth} (s): {min(values):.4e}")


class Benchmark(object):
    """
    Benchmark extension kernels
    """
    def __init__(self, subparser):
        from . import config
        parser = subparser.add_parser(
            'benchmark',
            parents=[config.parser],
            description=self.__doc__,
        )
        parser.add_argument(
            '-k', '--kernel',
            type=int,
            default=-1,
            choices=[-1]+list(range(len(KERNEL))),
            help=('Specify kernel to benchmark '
                  '(type: %(type)s, default: %(default)s, '
                  'choices: {%(choices)s})')
        )
        parser.add_argument(
            '-n',
            type=int,
            default=4096,
            help=('Number of particles '
                  '(type: %(type)s, default: %(default)s)')
        )
        parser.add_argument(
            '-r', '--rect',
            action='store_true',
            default=False,
            help=('Use rectangle kernel '
                  '(default: %(default)s)')
        )
        self.parser = parser

    def __call__(self, cli):
        k = cli.kernel
        n = cli.n
        rect = cli.rect
        backend = cli.backend
        if k == -1:
            for kernel in KERNEL:
                benchmark(kernel, n, backend, rect)
        else:
            benchmark(KERNEL[k], n, backend, rect)


# -- End of File --
