# -*- coding: utf-8 -*-
#

"""
A Python Toolkit for Astrophysical N-Body Simulations.
"""


from __future__ import print_function
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())


def main():
    """The top-level main function of tupan.

    """
    import os
    import sys
    import pprint
    import argparse

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
    from . import config
    from . import simulation
    simulation.add_parsers(subparser, parents=[config.parser, preparser])

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
