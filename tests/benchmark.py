# -*- coding: utf-8 -*-
#

"""

"""

import numpy as np
from tupan.particles.system import ParticleSystem
from tupan.lib import extensions as ext
from tupan.lib.utils.timing import Timer
from tupan.config import cli


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


def benchmark(bench, n_max=4096, backend=cli.backend):
    np.random.seed(0)

    name, kwargs = bench

    if backend == 'C':
        name += '_rectangle'

    kernel = ext.make_extension(name, backend)

    ips = set_particles(n_max)
    jps = set_particles(n_max)

    n = 2
    print('# benchmark:', name)
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


KERNEL = [
    ('Phi', {}),
    ('Acc', {}),
    ('AccJrk', {}),
    ('SnpCrk', {}),
    ('Tstep', {'eta': 1/64}),
    ('PNAcc', {'pn': {'order': 7, 'clight': 128.0}}),
    ('Sakura', {'dt': 1/64, 'flag': -2}),
]


benchmark(KERNEL[0])
benchmark(KERNEL[1])
benchmark(KERNEL[2])
benchmark(KERNEL[3])
benchmark(KERNEL[4])
benchmark(KERNEL[5])
benchmark(KERNEL[6])


# -- End of File --
