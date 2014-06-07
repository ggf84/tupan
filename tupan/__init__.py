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

    # create the top-level parser and subparser
    parser = argparse.ArgumentParser(description=__doc__)
    subparser = parser.add_subparsers(
        help="Consult specific help for details."
        )

    # add specific parsers to subparser
    from . import simulation
    simulation.add_parsers(subparser)

    # parse args
    args = parser.parse_args()
    print('#' * 25, file=sys.stderr)
    print(pprint.pformat(vars(args)), file=sys.stderr)
    print('#' * 25, file=sys.stderr)
    print('# PID:', os.getpid(), file=sys.stderr)
    print('#' * 25, file=sys.stderr)

    # set logging config
    level = getattr(logging, args.log.upper())
    format = "# %(asctime)s - %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(level=level, format=format)

    # run
    return args.func(args)


# -- End of File --
