#!/usr/bin/env python
# -*- coding: utf-8 -*-


if __name__ == "__main__":
    from pynbody import selftimer
    from pynbody.models import Plummer

    @selftimer
    def main():
        numBodies = 5471
        p = Plummer(numBodies, seed=1)
        p.make_plummer()
        p.write_snapshot()

        bi = p.bodies
        bi.calc_acc(bi)

    main()


########## end of file ##########
