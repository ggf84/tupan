# -*- coding: utf-8 -*-
#

"""
A Python Toolkit for Astrophysical N-Body Simulations.
"""


from __future__ import print_function
import argparse
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())


# create parent_parser and add arguments
parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument(
    '--backend',
    metavar='BACKEND',
    type=str,
    default='C',
    choices=['C', 'CL'],
    help=('Extension modules backend '
          '(type: %(type)s, default: %(default)s, '
          'choices: {%(choices)s}).')
    )
parent_parser.add_argument(
    '--fpwidth',
    metavar='FPWIDTH',
    type=str,
    default='fp64',
    choices=['fp32', 'fp64'],
    help=('Floating-point width '
          '(type: %(type)s, default: %(default)s, '
          'choices: {%(choices)s}).')
    )
parent_parser.add_argument(
    '--profile',
    action='store_true',
    help='Enable execution profile.'
    )
parent_parser.add_argument(
    '--view',
    action='store_true',
    help='Enable GLviewer support for visualization.'
    )

# parse known arguments from parent_parser
options, _ = parent_parser.parse_known_args()


def main():
    """The top-level main function of tupan.

    """
    import os
    import sys
    import pprint

    # create preparser and add arguments
    preparser = argparse.ArgumentParser(add_help=False)
    preparser.add_argument(
        '--log',
        metavar='LEVEL',
        type=str,
        default='critical',
        choices=[logging.getLevelName(i).lower() for i in range(0, 60, 10)],
        help=('Set the logging level '
              '(type: %(type)s, default: %(default)s, '
              'choices: {%(choices)s}).')
        )

    # set logging config
    args, _ = preparser.parse_known_args()
    level = getattr(logging, args.log.upper())
    fmt = '# %(asctime)s - %(levelname)s - %(name)s - %(message)s'
    logging.basicConfig(level=level, format=fmt)

    # create the main parser and subparser
    parser = argparse.ArgumentParser(description=__doc__)
    subparser = parser.add_subparsers(help='commands')

    # add specific parsers to subparser
    from . import simulation
    simulation.add_parsers(subparser, parents=[parent_parser, preparser])

    # parse args from the main parser
    args = parser.parse_args()
    print('#' * 25, file=sys.stderr)
    print(pprint.pformat(vars(args)), file=sys.stderr)
    print('#' * 25, file=sys.stderr)
    print('# PID:', os.getpid(), file=sys.stderr)
    print('#' * 25, file=sys.stderr)

    # run
    return args.func(args)


# -- End of File --
