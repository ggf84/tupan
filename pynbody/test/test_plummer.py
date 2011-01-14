#!/usr/bin/env python
# -*- coding: utf-8 -*-


if __name__ == "__main__":
    from pynbody.lib.decorators import timeit
    from pynbody.models import Plummer

    @timeit
    def main():
        numBodies = 16384
        p = Plummer(numBodies, seed=1)
        p.make_plummer()
#        p.dump_to_txt()

    main()


########## end of file ##########
