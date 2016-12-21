# -*- coding: utf-8 -*-
#

"""
Utility functions for benchmarking extension kernels.
"""

import numpy as np
from .particles.system import ParticleSystem
from .lib import extensions as ext
from .lib.utils.timing import Timer


KERNEL = [
    ('Phi', {}),
    ('Acc', {}),
    ('AccJrk', {}),
    ('SnpCrk', {}),
    ('Tstep', {'eta': 1/64}),
    ('PNAcc', {'pn': {'order': 7, 'clight': 128.0}}),
    ('Sakura', {'dt': 1/64, 'flag': -2}),
]


def best_of(n, func, *args, **kwargs):
    timer = Timer()
    elapsed = []
    for i in range(n):
        timer.start()
        func(*args, **kwargs)
        elapsed.append(timer.elapsed())
    return min(elapsed)


def set_particles(n):
    ps = ParticleSystem(n)
    ps.mass[...] = np.random.random((n,))
    ps.rdot[0][...] = np.random.random((3, n)) * 10
    ps.rdot[1][...] = np.random.random((3, n)) * 10
    ps.register_attribute('pnacc', '{nd}, {nb}', 'real_t')
    ps.register_attribute('drdot', '2, {nd}, {nb}', 'real_t')
    return ps


def benchmark(bench, n_max, backend):
    np.random.seed(0)

    name, kwargs = bench

    if backend == 'C':
        name += '_rectangle'

    if backend == 'CL' and name == 'Acc':
        name += '_rectangle'

    kernel = ext.make_extension(name, backend)

    ips = set_particles(n_max)
    jps = set_particles(n_max)

    n = 2
    print('\n# benchmark:', name)
    while n <= n_max:
        print('#    n:', n)
        elapsed = best_of(5, kernel.set_args, ips[:n], jps[:n], **kwargs)
        print("#        {meth} (s): {elapsed:.4e}"
              .format(meth='set', elapsed=elapsed))
        elapsed = best_of(3, kernel.run)
        print("#        {meth} (s): {elapsed:.4e}"
              .format(meth='run', elapsed=elapsed))
        elapsed = best_of(5, kernel.map_buffers)
        print("#        {meth} (s): {elapsed:.4e}"
              .format(meth='get', elapsed=elapsed))
        n *= 2


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
            '-n', '--n_max',
            type=int,
            default=4096,
            help=('Max number of particles '
                  '(type: %(type)s, default: %(default)s)')
        )
        self.parser = parser

    def __call__(self, cli):
        k = cli.kernel
        n_max = cli.n_max
        backend = cli.backend
        if k == -1:
            for kernel in KERNEL:
                benchmark(kernel, n_max, backend)
        else:
            benchmark(KERNEL[k], n_max, backend)


# -- End of File --
