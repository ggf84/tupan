# -*- coding: utf-8 -*-
#

"""
A python toolkit for astrophysical N-body simulations
"""

from __future__ import print_function
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())


def main():
    import os
    import sys
    import pprint
    import argparse

    # create parser
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    subparser = parser.add_subparsers(
        dest='command',
        help='command help',
    )

    # setup 'run'
    from .simulation import Run
    run = Run(subparser)

    # setup 'restart'
    from .simulation import Restart
    restart = Restart(subparser)

    # setup 'benchmark'
    from .benchmarks import Benchmark
    benchmark = Benchmark(subparser)

    # parse arguments
    cli = parser.parse_args()
    if cli.command is None:
        return parser.print_usage()

    print('#' * 25, file=sys.stderr)
    print(pprint.pformat(vars(cli)), file=sys.stderr)
    print('#' * 25, file=sys.stderr)
    print('# PID:', os.getpid(), file=sys.stderr)
    print('#' * 25, file=sys.stderr)

    # finally, run the command!
    if cli.command == 'run':
        return run(cli)
    if cli.command == 'restart':
        return restart(cli)
    if cli.command == 'benchmark':
        return benchmark(cli)


# -- End of File --
