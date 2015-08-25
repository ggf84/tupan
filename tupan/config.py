# -*- coding: utf-8 -*-
#

"""
TODO.
"""

import os
import sys
import getpass
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
          'choices: {%(choices)s}).')
    )
parser.add_argument(
    '--cache_prefix',
    type=str,
    default=os.path.expanduser('~'),
    help=('Cache dir prefix '
          '(type: %(type)s, default: %(default)s).')
    )
parser.add_argument(
    '--fpwidth',
    metavar='FPWIDTH',
    type=str,
    default='fp64',
    choices=['fp32', 'fp64'],
    help=('Floating-point width '
          '(type: %(type)s, default: %(default)s, '
          'choices: {%(choices)s}).')
    )
parser.add_argument(
    '--profile',
    action='store_true',
    help='Enable execution profile.'
    )
parser.add_argument(
    '--view',
    metavar='FREQ',
    nargs='?',
    type=int,
    const=1,
    default=0,
    help=('Enable support for real-time visualization. '
          'Optionally you can pass an update frequency '
          '(type: %(type)s, const: %(const)s, default: %(default)s).')
    )
parser.add_argument(
    '--record',
    action='store_true',
    help='Enable recording of visualization.'
    )

# parse known arguments from parser
options, _ = parser.parse_known_args()


def get_cache_dir(*postfix):
    user = getpass.getuser()
    sys_version = '.'.join(str(i) for i in sys.version_info)
    basename = '.tupan-cache-uid{0}-py{1}'.format(user, sys_version)
    prefix = options.cache_prefix
    cache_dir = os.path.abspath(os.path.join(prefix, basename, *postfix))
    try:
        os.makedirs(cache_dir)
    except OSError:
        pass
    return cache_dir


# -- End of File --
