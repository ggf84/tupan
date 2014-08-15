# -*- coding: utf-8 -*-
#

"""
TODO.
"""


import logging
from .base import Base, power_of_two
from ..lib.utils.timing import timings, bind_all


__all__ = ['Hermite']

LOGGER = logging.getLogger(__name__)


class HX(object):
    """

    """
    order = None

    def __init__(self, manager):
        self.initialized = False
        self.manager = manager

    @staticmethod
    def prepare(ps, eta):
        raise NotImplementedError

    @staticmethod
    def predict(ps, dt):
        raise NotImplementedError

    @staticmethod
    def ecorrect(ps1, ps0, dt):
        raise NotImplementedError

    @staticmethod
    def set_nextstep(ps, eta):
        raise NotImplementedError

    def pec(self, n, ps, eta, dtmax):
        if not self.initialized:
            self.initialized = True
            ps = self.prepare(ps, eta)
            if self.order > 4:
                n += 1
        dt = power_of_two(ps, dtmax) if self.manager.update_tstep else dtmax
        ps0 = ps.copy()
        ps1 = self.predict(ps, dt)
        while n > 0:
            ps1 = self.ecorrect(ps1, ps0, dt)
            n -= 1
        type(ps1).t_curr += dt
        ps1.tstep[...] = dt
        ps1.time += dt
        ps1.nstep += 1
        self.manager.dump(dt, ps)
        if self.manager.update_tstep:
            self.set_nextstep(ps1, eta)
        return ps1


@bind_all(timings)
class H2(HX):
    """

    """
    order = 2

    @staticmethod
    def prepare(ps, eta):
        ps.set_tstep(ps, eta)
        ps.set_acc(ps)
        ps.jx = ps.vx * 0
        ps.jy = ps.vy * 0
        ps.jz = ps.vz * 0
        return ps

    @staticmethod
    def predict(ps, dt):
        """

        """
        h = dt

        ps.rx += h * (ps.vx)
        ps.ry += h * (ps.vy)
        ps.rz += h * (ps.vz)

        return ps

    @staticmethod
    def ecorrect(ps1, ps0, dt):
        """

        """
        h = dt
        h2 = h / 2

        ps1.set_acc(ps1)

        ax_p = (ps0.ax + ps1.ax)
        ay_p = (ps0.ay + ps1.ay)
        az_p = (ps0.az + ps1.az)

        ps1.vx[...] = ps0.vx + h2 * (ax_p)
        ps1.vy[...] = ps0.vy + h2 * (ay_p)
        ps1.vz[...] = ps0.vz + h2 * (az_p)

        vx_p = (ps0.vx + ps1.vx)
        vy_p = (ps0.vy + ps1.vy)
        vz_p = (ps0.vz + ps1.vz)

        ps1.rx[...] = ps0.rx + h2 * (vx_p)
        ps1.ry[...] = ps0.ry + h2 * (vy_p)
        ps1.rz[...] = ps0.rz + h2 * (vz_p)

        hinv = 1 / h

        ax_m = (ps0.ax - ps1.ax)
        ay_m = (ps0.ay - ps1.ay)
        az_m = (ps0.az - ps1.az)

        jx = -hinv * (ax_m)
        jy = -hinv * (ay_m)
        jz = -hinv * (az_m)

        ps1.jx[...] = jx
        ps1.jy[...] = jy
        ps1.jz[...] = jz

        return ps1

    @staticmethod
    def set_nextstep(ps, eta):
        s0 = (ps.ax**2 + ps.ay**2 + ps.az**2)
        s1 = (ps.jx**2 + ps.jy**2 + ps.jz**2)

        u = s0
        l = s1

        ps.tstep[...] = eta * (u / l)**0.5


@bind_all(timings)
class H4(HX):
    """

    """
    order = 4

    @staticmethod
    def prepare(ps, eta):
        ps.set_tstep(ps, eta)
        ps.set_acc_jerk(ps)
        ps.sx = ps.ax * 0
        ps.sy = ps.ay * 0
        ps.sz = ps.az * 0
        ps.cx = ps.jx * 0
        ps.cy = ps.jy * 0
        ps.cz = ps.jz * 0
        return ps

    @staticmethod
    def predict(ps, dt):
        """

        """
        h = dt
        h2 = h / 2
        h3 = h / 3

        ps.rx += h * (ps.vx + h2 * (ps.ax + h3 * (ps.jx)))
        ps.ry += h * (ps.vy + h2 * (ps.ay + h3 * (ps.jy)))
        ps.rz += h * (ps.vz + h2 * (ps.az + h3 * (ps.jz)))

        ps.vx += h * (ps.ax + h2 * (ps.jx))
        ps.vy += h * (ps.ay + h2 * (ps.jy))
        ps.vz += h * (ps.az + h2 * (ps.jz))

        return ps

    @staticmethod
    def ecorrect(ps1, ps0, dt):
        """

        """
        h = dt
        h2 = h / 2
        h6 = h / 6

        ps1.set_acc_jerk(ps1)

        jx_m = (ps0.jx - ps1.jx)
        jy_m = (ps0.jy - ps1.jy)
        jz_m = (ps0.jz - ps1.jz)
        ax_p = (ps0.ax + ps1.ax)
        ay_p = (ps0.ay + ps1.ay)
        az_p = (ps0.az + ps1.az)

        ps1.vx[...] = ps0.vx + h2 * (ax_p + h6 * (jx_m))
        ps1.vy[...] = ps0.vy + h2 * (ay_p + h6 * (jy_m))
        ps1.vz[...] = ps0.vz + h2 * (az_p + h6 * (jz_m))

        ax_m = (ps0.ax - ps1.ax)
        ay_m = (ps0.ay - ps1.ay)
        az_m = (ps0.az - ps1.az)
        vx_p = (ps0.vx + ps1.vx)
        vy_p = (ps0.vy + ps1.vy)
        vz_p = (ps0.vz + ps1.vz)

        ps1.rx[...] = ps0.rx + h2 * (vx_p + h6 * (ax_m))
        ps1.ry[...] = ps0.ry + h2 * (vy_p + h6 * (ay_m))
        ps1.rz[...] = ps0.rz + h2 * (vz_p + h6 * (az_m))

        hinv = 1 / h
        hinv2 = hinv * hinv
        hinv3 = hinv * hinv2

        jx_p = (ps0.jx + ps1.jx)
        jy_p = (ps0.jy + ps1.jy)
        jz_p = (ps0.jz + ps1.jz)

        sx = -hinv * (jx_m)
        sy = -hinv * (jy_m)
        sz = -hinv * (jz_m)
        cx = 6 * hinv3 * (2 * ax_m + h * jx_p)
        cy = 6 * hinv3 * (2 * ay_m + h * jy_p)
        cz = 6 * hinv3 * (2 * az_m + h * jz_p)
        sx += h2 * (cx)
        sy += h2 * (cy)
        sz += h2 * (cz)

        ps1.sx[...] = sx
        ps1.sy[...] = sy
        ps1.sz[...] = sz
        ps1.cx[...] = cx
        ps1.cy[...] = cy
        ps1.cz[...] = cz

        return ps1

    @staticmethod
    def set_nextstep(ps, eta):
        s0 = (ps.ax**2 + ps.ay**2 + ps.az**2)
        s1 = (ps.jx**2 + ps.jy**2 + ps.jz**2)
        s2 = (ps.sx**2 + ps.sy**2 + ps.sz**2)
        s3 = (ps.cx**2 + ps.cy**2 + ps.cz**2)

        u = (s0 * s2)**0.5 + s1
        l = (s1 * s3)**0.5 + s2

        ps.tstep[...] = eta * (u / l)**0.5


@bind_all(timings)
class H6(HX):
    """

    """
    order = 6

    @staticmethod
    def prepare(ps, eta):
        ps.set_tstep(ps, eta)
        ps.set_acc_jerk(ps)
        ps.set_snap_crackle(ps)
        ps.cx = ps.jx * 0
        ps.cy = ps.jy * 0
        ps.cz = ps.jz * 0
        ps.d4ax = ps.sx * 0
        ps.d4ay = ps.sy * 0
        ps.d4az = ps.sz * 0
        ps.d5ax = ps.cx * 0
        ps.d5ay = ps.cy * 0
        ps.d5az = ps.cz * 0
        return ps

    @staticmethod
    def predict(ps, dt):
        """

        """
        h = dt
        h2 = h / 2
        h3 = h / 3
        h4 = h / 4
        h5 = h / 5

        ps.rx += h * (ps.vx
                      + h2 * (ps.ax
                              + h3 * (ps.jx
                                      + h4 * (ps.sx
                                              + h5 * (ps.cx)))))
        ps.ry += h * (ps.vy
                      + h2 * (ps.ay
                              + h3 * (ps.jy
                                      + h4 * (ps.sy
                                              + h5 * (ps.cy)))))
        ps.rz += h * (ps.vz
                      + h2 * (ps.az
                              + h3 * (ps.jz
                                      + h4 * (ps.sz
                                              + h5 * (ps.cz)))))

        ps.vx += h * (ps.ax + h2 * (ps.jx + h3 * (ps.sx + h4 * (ps.cx))))
        ps.vy += h * (ps.ay + h2 * (ps.jy + h3 * (ps.sy + h4 * (ps.cy))))
        ps.vz += h * (ps.az + h2 * (ps.jz + h3 * (ps.sz + h4 * (ps.cz))))

        ps.ax += h * (ps.jx + h2 * (ps.sx + h3 * (ps.cx)))
        ps.ay += h * (ps.jy + h2 * (ps.sy + h3 * (ps.cy)))
        ps.az += h * (ps.jz + h2 * (ps.sz + h3 * (ps.cz)))

        return ps

    @staticmethod
    def ecorrect(ps1, ps0, dt):
        """

        """
        h = dt
        h2 = h / 2
        h5 = h / 5
        h12 = h / 12

        ps1.set_snap_crackle(ps1)
        ps1.set_acc_jerk(ps1)

        sx_p = (ps0.sx + ps1.sx)
        sy_p = (ps0.sy + ps1.sy)
        sz_p = (ps0.sz + ps1.sz)
        jx_m = (ps0.jx - ps1.jx)
        jy_m = (ps0.jy - ps1.jy)
        jz_m = (ps0.jz - ps1.jz)
        ax_p = (ps0.ax + ps1.ax)
        ay_p = (ps0.ay + ps1.ay)
        az_p = (ps0.az + ps1.az)

        ps1.vx[...] = ps0.vx + h2 * (ax_p + h5 * (jx_m + h12 * (sx_p)))
        ps1.vy[...] = ps0.vy + h2 * (ay_p + h5 * (jy_m + h12 * (sy_p)))
        ps1.vz[...] = ps0.vz + h2 * (az_p + h5 * (jz_m + h12 * (sz_p)))

        jx_p = (ps0.jx + ps1.jx)
        jy_p = (ps0.jy + ps1.jy)
        jz_p = (ps0.jz + ps1.jz)
        ax_m = (ps0.ax - ps1.ax)
        ay_m = (ps0.ay - ps1.ay)
        az_m = (ps0.az - ps1.az)
        vx_p = (ps0.vx + ps1.vx)
        vy_p = (ps0.vy + ps1.vy)
        vz_p = (ps0.vz + ps1.vz)

        ps1.rx[...] = ps0.rx + h2 * (vx_p + h5 * (ax_m + h12 * (jx_p)))
        ps1.ry[...] = ps0.ry + h2 * (vy_p + h5 * (ay_m + h12 * (jy_p)))
        ps1.rz[...] = ps0.rz + h2 * (vz_p + h5 * (az_m + h12 * (jz_p)))

        hinv = 1 / h
        hinv2 = hinv * hinv
        hinv3 = hinv * hinv2
        hinv5 = hinv2 * hinv3

        sx_m = (ps0.sx - ps1.sx)
        sy_m = (ps0.sy - ps1.sy)
        sz_m = (ps0.sz - ps1.sz)

        cx = 3 * hinv3 * (10 * ax_m + h * (5 * jx_p + h2 * sx_m))
        cy = 3 * hinv3 * (10 * ay_m + h * (5 * jy_p + h2 * sy_m))
        cz = 3 * hinv3 * (10 * az_m + h * (5 * jz_p + h2 * sz_m))
        d4ax = 6 * hinv3 * (2 * jx_m + h * sx_p)
        d4ay = 6 * hinv3 * (2 * jy_m + h * sy_p)
        d4az = 6 * hinv3 * (2 * jz_m + h * sz_p)
        d5ax = -60 * hinv5 * (12 * ax_m + h * (6 * jx_p + h * sx_m))
        d5ay = -60 * hinv5 * (12 * ay_m + h * (6 * jy_p + h * sy_m))
        d5az = -60 * hinv5 * (12 * az_m + h * (6 * jz_p + h * sz_m))

        h4 = h / 4
        cx += h2 * (d4ax + h4 * d5ax)
        cy += h2 * (d4ay + h4 * d5ay)
        cz += h2 * (d4az + h4 * d5az)
        d4ax += h2 * (d5ax)
        d4ay += h2 * (d5ay)
        d4az += h2 * (d5az)

        ps1.cx[...] = cx
        ps1.cy[...] = cy
        ps1.cz[...] = cz

        ps1.d4ax[...] = d4ax
        ps1.d4ay[...] = d4ay
        ps1.d4az[...] = d4az
        ps1.d5ax[...] = d5ax
        ps1.d5ay[...] = d5ay
        ps1.d5az[...] = d5az

        return ps1

    @staticmethod
    def set_nextstep(ps, eta):
        s0 = (ps.ax**2 + ps.ay**2 + ps.az**2)
        s1 = (ps.jx**2 + ps.jy**2 + ps.jz**2)
        s2 = (ps.sx**2 + ps.sy**2 + ps.sz**2)
        s3 = (ps.cx**2 + ps.cy**2 + ps.cz**2)
        s4 = (ps.d4ax**2 + ps.d4ay**2 + ps.d4az**2)
        s5 = (ps.d5ax**2 + ps.d5ay**2 + ps.d5az**2)

        u = (s0 * s2)**0.5 + s1
        l = (s3 * s5)**0.5 + s4

        ps.tstep[...] = eta * (u / l)**(1.0 / 6)


@bind_all(timings)
class H8(HX):
    """

    """
    order = 8

    @staticmethod
    def prepare(ps, eta):
        ps.set_tstep(ps, eta)
        ps.set_acc_jerk(ps)
        ps.set_snap_crackle(ps)
        ps.d4ax = ps.sx * 0
        ps.d4ay = ps.sy * 0
        ps.d4az = ps.sz * 0
        ps.d5ax = ps.cx * 0
        ps.d5ay = ps.cy * 0
        ps.d5az = ps.cz * 0
        ps.d6ax = ps.d4ax * 0
        ps.d6ay = ps.d4ay * 0
        ps.d6az = ps.d4az * 0
        ps.d7ax = ps.d5ax * 0
        ps.d7ay = ps.d5ay * 0
        ps.d7az = ps.d5az * 0
        return ps

    @staticmethod
    def predict(ps, dt):
        """

        """
        h = dt
        h2 = h / 2
        h3 = h / 3
        h4 = h / 4
        h5 = h / 5
        h6 = h / 6
        h7 = h / 7

        ps.rx += h * (ps.vx
                      + h2 * (ps.ax
                              + h3 * (ps.jx
                                      + h4 * (ps.sx
                                              + h5 * (ps.cx
                                                      + h6 * (ps.d4ax
                                                              + h7 * (ps.d5ax)))))))
        ps.ry += h * (ps.vy
                      + h2 * (ps.ay
                              + h3 * (ps.jy
                                      + h4 * (ps.sy
                                              + h5 * (ps.cy
                                                      + h6 * (ps.d4ay
                                                              + h7 * (ps.d5ay)))))))
        ps.rz += h * (ps.vz
                      + h2 * (ps.az
                              + h3 * (ps.jz
                                      + h4 * (ps.sz
                                              + h5 * (ps.cz
                                                      + h6 * (ps.d4az
                                                              + h7 * (ps.d5az)))))))

        ps.vx += h * (ps.ax
                      + h2 * (ps.jx
                              + h3 * (ps.sx
                                      + h4 * (ps.cx
                                              + h5 * (ps.d4ax
                                                      + h6 * (ps.d5ax))))))
        ps.vy += h * (ps.ay
                      + h2 * (ps.jy
                              + h3 * (ps.sy
                                      + h4 * (ps.cy
                                              + h5 * (ps.d4ay
                                                      + h6 * (ps.d5ay))))))
        ps.vz += h * (ps.az
                      + h2 * (ps.jz
                              + h3 * (ps.sz
                                      + h4 * (ps.cz
                                              + h5 * (ps.d4az
                                                      + h6 * (ps.d5az))))))

        ps.ax += h * (ps.jx
                      + h2 * (ps.sx
                              + h3 * (ps.cx
                                      + h4 * (ps.d4ax
                                              + h5 * (ps.d5ax)))))
        ps.ay += h * (ps.jy
                      + h2 * (ps.sy
                              + h3 * (ps.cy
                                      + h4 * (ps.d4ay
                                              + h5 * (ps.d5ay)))))
        ps.az += h * (ps.jz
                      + h2 * (ps.sz
                              + h3 * (ps.cz
                                      + h4 * (ps.d4az
                                              + h5 * (ps.d5az)))))

        ps.jx += h * (ps.sx + h2 * (ps.cx + h3 * (ps.d4ax + h4 * (ps.d5ax))))
        ps.jy += h * (ps.sy + h2 * (ps.cy + h3 * (ps.d4ay + h4 * (ps.d5ay))))
        ps.jz += h * (ps.sz + h2 * (ps.cz + h3 * (ps.d4az + h4 * (ps.d5az))))

        return ps

    @staticmethod
    def ecorrect(ps1, ps0, dt):
        """

        """
        h = dt
        h2 = h / 2
        h3 = h / 3
        h14 = h / 14
        h20 = h / 20

        ps1.set_snap_crackle(ps1)
        ps1.set_acc_jerk(ps1)

        cx_m = (ps0.cx - ps1.cx)
        cy_m = (ps0.cy - ps1.cy)
        cz_m = (ps0.cz - ps1.cz)
        sx_p = (ps0.sx + ps1.sx)
        sy_p = (ps0.sy + ps1.sy)
        sz_p = (ps0.sz + ps1.sz)
        jx_m = (ps0.jx - ps1.jx)
        jy_m = (ps0.jy - ps1.jy)
        jz_m = (ps0.jz - ps1.jz)
        ax_p = (ps0.ax + ps1.ax)
        ay_p = (ps0.ay + ps1.ay)
        az_p = (ps0.az + ps1.az)

        ps1.vx[...] = ps0.vx + h2 * (ax_p
                                     + h14 * (3 * jx_m
                                              + h3 * (sx_p
                                                      + h20 * (cx_m))))
        ps1.vy[...] = ps0.vy + h2 * (ay_p
                                     + h14 * (3 * jy_m
                                              + h3 * (sy_p
                                                      + h20 * (cy_m))))
        ps1.vz[...] = ps0.vz + h2 * (az_p
                                     + h14 * (3 * jz_m
                                              + h3 * (sz_p
                                                      + h20 * (cz_m))))

        sx_m = (ps0.sx - ps1.sx)
        sy_m = (ps0.sy - ps1.sy)
        sz_m = (ps0.sz - ps1.sz)
        jx_p = (ps0.jx + ps1.jx)
        jy_p = (ps0.jy + ps1.jy)
        jz_p = (ps0.jz + ps1.jz)
        ax_m = (ps0.ax - ps1.ax)
        ay_m = (ps0.ay - ps1.ay)
        az_m = (ps0.az - ps1.az)
        vx_p = (ps0.vx + ps1.vx)
        vy_p = (ps0.vy + ps1.vy)
        vz_p = (ps0.vz + ps1.vz)

        ps1.rx[...] = ps0.rx + h2 * (vx_p
                                     + h14 * (3 * ax_m
                                              + h3 * (jx_p
                                                      + h20 * (sx_m))))
        ps1.ry[...] = ps0.ry + h2 * (vy_p
                                     + h14 * (3 * ay_m
                                              + h3 * (jy_p
                                                      + h20 * (sy_m))))
        ps1.rz[...] = ps0.rz + h2 * (vz_p
                                     + h14 * (3 * az_m
                                              + h3 * (jz_p
                                                      + h20 * (sz_m))))

        hinv = 1 / h
        hinv2 = hinv * hinv
        hinv3 = hinv * hinv2
        hinv5 = hinv2 * hinv3
        hinv7 = hinv2 * hinv5

        cx_p = (ps0.cx + ps1.cx)
        cy_p = (ps0.cy + ps1.cy)
        cz_p = (ps0.cz + ps1.cz)

        d4ax = 3 * hinv3 * (10 * jx_m + h * (5 * sx_p + h2 * cx_m))
        d4ay = 3 * hinv3 * (10 * jy_m + h * (5 * sy_p + h2 * cy_m))
        d4az = 3 * hinv3 * (10 * jz_m + h * (5 * sz_p + h2 * cz_m))
        d5ax = -15 * hinv5 * (168 * ax_m
                              + h * (84 * jx_p
                                     + h * (16 * sx_m
                                            + h * cx_p)))
        d5ay = -15 * hinv5 * (168 * ay_m
                              + h * (84 * jy_p
                                     + h * (16 * sy_m
                                            + h * cy_p)))
        d5az = -15 * hinv5 * (168 * az_m
                              + h * (84 * jz_p
                                     + h * (16 * sz_m
                                            + h * cz_p)))
        d6ax = -60 * hinv5 * (12 * jx_m + h * (6 * sx_p + h * cx_m))
        d6ay = -60 * hinv5 * (12 * jy_m + h * (6 * sy_p + h * cy_m))
        d6az = -60 * hinv5 * (12 * jz_m + h * (6 * sz_p + h * cz_m))
        d7ax = 840 * hinv7 * (120 * ax_m
                              + h * (60 * jx_p
                                     + h * (12 * sx_m
                                            + h * cx_p)))
        d7ay = 840 * hinv7 * (120 * ay_m
                              + h * (60 * jy_p
                                     + h * (12 * sy_m
                                            + h * cy_p)))
        d7az = 840 * hinv7 * (120 * az_m
                              + h * (60 * jz_p
                                     + h * (12 * sz_m
                                            + h * cz_p)))

        h4 = h / 4
        h6 = h / 6
        d4ax += h2 * (d5ax + h4 * (d6ax + h6 * d7ax))
        d4ay += h2 * (d5ay + h4 * (d6ay + h6 * d7ay))
        d4az += h2 * (d5az + h4 * (d6az + h6 * d7az))
        d5ax += h2 * (d6ax + h4 * d7ax)
        d5ay += h2 * (d6ay + h4 * d7ay)
        d5az += h2 * (d6az + h4 * d7az)
        d6ax += h2 * (d7ax)
        d6ay += h2 * (d7ay)
        d6az += h2 * (d7az)

        ps1.d4ax[...] = d4ax
        ps1.d4ay[...] = d4ay
        ps1.d4az[...] = d4az
        ps1.d5ax[...] = d5ax
        ps1.d5ay[...] = d5ay
        ps1.d5az[...] = d5az

        ps1.d6ax[...] = d6ax
        ps1.d6ay[...] = d6ay
        ps1.d6az[...] = d6az
        ps1.d7ax[...] = d7ax
        ps1.d7ay[...] = d7ay
        ps1.d7az[...] = d7az

        return ps1

    @staticmethod
    def set_nextstep(ps, eta):
        s0 = (ps.ax**2 + ps.ay**2 + ps.az**2)
        s1 = (ps.jx**2 + ps.jy**2 + ps.jz**2)
        s2 = (ps.sx**2 + ps.sy**2 + ps.sz**2)
#        s3 = (ps.cx**2 + ps.cy**2 + ps.cz**2)
#        s4 = (ps.d4ax**2 + ps.d4ay**2 + ps.d4az**2)
        s5 = (ps.d5ax**2 + ps.d5ay**2 + ps.d5az**2)
        s6 = (ps.d6ax**2 + ps.d6ay**2 + ps.d6az**2)
        s7 = (ps.d7ax**2 + ps.d7ay**2 + ps.d7az**2)

        u = (s0 * s2)**0.5 + s1
        l = (s5 * s7)**0.5 + s6

        ps.tstep[...] = eta * (u / l)**(1.0 / 10)


@bind_all(timings)
class Hermite(Base):
    """

    """
    PROVIDED_METHODS = [
        'hermite2c', 'hermite2a',
        'hermite4c', 'hermite4a',
        'hermite6c', 'hermite6a',
        'hermite8c', 'hermite8a',
    ]

    def __init__(self, eta, time, ps, method, **kwargs):
        """

        """
        super(Hermite, self).__init__(eta, time, ps, **kwargs)
        self.method = method

        if method not in self.PROVIDED_METHODS:
            raise ValueError('Invalid integration method: {0}'.format(method))

        if method.endswith('c'):
            self.update_tstep = False
        elif method.endswith('a'):
            self.update_tstep = True

        if 'hermite2' in method:
            self.hermite = H2(self)
        elif 'hermite4' in method:
            self.hermite = H4(self)
        elif 'hermite6' in method:
            self.hermite = H6(self)
        elif 'hermite8' in method:
            self.hermite = H8(self)

    def initialize(self, t_end):
        """

        """
        ps = self.ps
        LOGGER.info("Initializing '%s' integrator at "
                    "t_curr = %g and t_end = %g.",
                    self.method, ps.t_curr, t_end)

        if self.reporter:
            self.reporter.diagnostic_report(ps)
        if self.dumpper:
            self.dumpper.dump_worldline(ps)
        if self.viewer:
            self.viewer.show_event(ps)

        self.is_initialized = True

    def finalize(self, t_end):
        """

        """
        ps = self.ps
        LOGGER.info("Finalizing '%s' integrator at "
                    "t_curr = %g and t_end = %g.",
                    self.method, ps.t_curr, t_end)

        if self.viewer:
            self.viewer.show_event(ps)
            self.viewer.enter_main_loop()

    def do_step(self, ps, dtmax):
        """

        """
        return self.hermite.pec(2, ps, self.eta, dtmax)


# -- End of File --
