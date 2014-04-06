#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

"""
Use this script to perform a new N-body simulation,
or restart a simulation from a previous run.

For details about command line arguments:

    $ python tupan-simulation.py --help
    $ python tupan-simulation.py newrun --help
    $ python tupan-simulation.py restart --help

"""


if __name__ == "__main__":
    from tupan import simulation
    simulation.main()


# -- End of File --
