# -*- coding: utf-8 -*-
#

"""
TODO.
"""

import logging
import argparse


# create parser and add arguments
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '--backend',
    metavar='BACKEND',
    type=str,
    default='C',
    choices=['C', 'CL'],
    help=('Extension modules backend '
          '(type: %(type)s, default: %(default)s, '
          'choices: {%(choices)s})')
)
parser.add_argument(
    '--fpwidth',
    metavar='FPWIDTH',
    type=str,
    default='fp64',
    choices=['fp32', 'fp64'],
    help=('Floating-point width '
          '(type: %(type)s, default: %(default)s, '
          'choices: {%(choices)s})')
)
parser.add_argument(
    '--view',
    metavar='NSTEPS',
    nargs='?',
    type=int,
    const=1,
    help=('Enable real-time visualization. NSTEPS is the '
          '(const: %(const)s) number of time-steps between '
          'updates (type: %(type)s, default: %(default)s)')
)
parser.add_argument(
    '--record',
    metavar='FPS',
    nargs='?',
    type=int,
    const=24,
    help=('Record visualization at a given (const: %(const)s) '
          'FPS ratio (type: %(type)s, default: %(default)s)')
)
parser.add_argument(
    '--log',
    metavar='LEVEL',
    type=str,
    default='critical',
    choices=[logging.getLevelName(i).lower() for i in range(0, 60, 10)],
    help=('Set the logging level '
          '(type: %(type)s, default: %(default)s, '
          'choices: {%(choices)s})')
)

# parse known arguments from parser
cli, _ = parser.parse_known_args()

# set logging config
level = getattr(logging, cli.log.upper())
fmt = '# %(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(level=level, format=fmt)


# -- End of File --
