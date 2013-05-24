#include "common.h"


#ifdef DOUBLE
    #define TOLERANCE ((REAL)2.3283064365386962891e-10)     // sqrt(2^-64)
#else
    #define TOLERANCE ((REAL)1.52587890625e-5)              // sqrt(2^-32)
#endif
#define MAXITER 64
#define COMPARE(x, y) (((x) > (y)) - ((x) < (y)))
#define SIGN(x) COMPARE(x, 0)


inline REAL stumpff_c0(
    const REAL zeta)
{
    if (zeta > 0) {
        REAL sz = sqrt(zeta);
        return cos(sz);
    }
    if (zeta < 0) {
        REAL sz = sqrt(-zeta);
        return cosh(sz);
    }
    return ((REAL)1);
}


inline REAL stumpff_c1(
    const REAL zeta)
{
    if (zeta > 0) {
        REAL sz = sqrt(zeta);
        return sin(sz) / sz;
    }
    if (zeta < 0) {
        REAL sz = sqrt(-zeta);
        return sinh(sz) / sz;
    }
    return ((REAL)1);
}


inline REAL stumpff_c2(
    const REAL zeta)
{
    if (zeta > 0) {
        REAL sz = sqrt(zeta);
        return (1 - cos(sz)) / zeta;
    }
    if (zeta < 0) {
        REAL sz = sqrt(-zeta);
        return (1 - cosh(sz)) / zeta;
    }
    return 1/((REAL)2);
}


inline REAL stumpff_c3(
    const REAL zeta)
{
    if (zeta > 0) {
        REAL sz = sqrt(zeta);
        return (1 - sin(sz) / sz) / zeta;
    }
    if (zeta < 0) {
        REAL sz = sqrt(-zeta);
        return (1 - sinh(sz) / sz) / zeta;
    }
    return 1/((REAL)6);
}


inline REAL S0(
    const REAL s,
    const REAL alpha)
{
    REAL s2 = s * s;
    REAL zeta = alpha * s2;
    return stumpff_c0(zeta);
}


inline REAL S1(
    const REAL s,
    const REAL alpha)
{
    REAL s2 = s * s;
    REAL zeta = alpha * s2;
    return s * stumpff_c1(zeta);
}


inline REAL S2(
    const REAL s,
    const REAL alpha)
{
    REAL s2 = s * s;
    REAL zeta = alpha * s2;
    return s2 * stumpff_c2(zeta);
}


inline REAL S3(
    const REAL s,
    const REAL alpha)
{
    REAL s2 = s * s;
    REAL s3 = s * s2;
    REAL zeta = alpha * s2;
    return s3 * stumpff_c3(zeta);
}


inline REAL lagrange_f(
    const REAL s,
    const REAL r0,
    const REAL m,
    const REAL alpha)
{
    REAL m_r0 = m / r0;
    return 1 - m_r0 * S2(s, alpha);
}


inline REAL lagrange_dfds(
    const REAL s,
    const REAL r0,
    const REAL r1,
    const REAL m,
    const REAL alpha)
{
    REAL m_r0r1 = m / (r0 * r1);
    return -m_r0r1 * S1(s, alpha);
}


inline REAL lagrange_g(
    const REAL dt,
    const REAL s,
    const REAL m,
    const REAL alpha)
{
    return dt - m * S3(s, alpha);
}


inline REAL lagrange_dgds(
    const REAL s,
    const REAL r1,
    const REAL m,
    const REAL alpha)
{
    REAL m_r1 = m / r1;
    return 1 - m_r1 * S2(s, alpha);
}


inline REAL universal_kepler(
    const REAL s,
    const REAL r0,
    const REAL r0v0,
    const REAL m,
    const REAL alpha)
{
    return r0 * S1(s, alpha) + r0v0 * S2(s, alpha) + m * S3(s, alpha);
}


inline REAL universal_kepler_ds(
    const REAL s,
    const REAL r0,
    const REAL r0v0,
    const REAL m,
    const REAL alpha)
{
    return r0 * S0(s, alpha) + r0v0 * S1(s, alpha) + m * S2(s, alpha);
}


inline REAL universal_kepler_dsds(
    const REAL s,
    const REAL r0,
    const REAL r0v0,
    const REAL m,
    const REAL alpha)
{
    return (m - alpha * r0) * S1(s, alpha) + r0v0 * S0(s, alpha);
}


inline REAL f(
    const REAL s,
    REAL *arg)
{
    return universal_kepler(s, arg[1], arg[2], arg[3], arg[4]) - arg[0];
}


inline REAL fprime(
    const REAL s,
    REAL *arg)
{
    return universal_kepler_ds(s, arg[1], arg[2], arg[3], arg[4]);
}


inline REAL fprimeprime(
    const REAL s,
    REAL *arg)
{
    return universal_kepler_dsds(s, arg[1], arg[2], arg[3], arg[4]);
}


#define ORDER 5
inline int laguerre(
    REAL x0,
    REAL *x,
    REAL *arg)
{
    int i = 0;
    REAL delta;

    *x = x0;
    do {
        REAL fv = f(*x, arg);
        REAL dfv = fprime(*x, arg);
        REAL ddfv = fprimeprime(*x, arg);

        REAL a = dfv;
        REAL a2 = a * a;
        REAL b = a2 - fv * ddfv;
        REAL g = ORDER * fv;
        REAL h = a + SIGN(a) * sqrt(fabs((ORDER - 1) * (ORDER * b - a2)));
        if (h == 0) return -1;
        delta = -g / h;

        (*x) += delta;
        i += 1;
        if (i > MAXITER) return -2;
    } while (fabs(delta) > TOLERANCE);
    if (SIGN((*x)) != SIGN(x0)) return -3;
    return 0;
}


inline int halley(
    REAL x0,
    REAL *x,
    REAL *arg)
{
    int i = 0;
    REAL delta;

    *x = x0;
    do {
        REAL fv = f(*x, arg);
        REAL dfv = fprime(*x, arg);
        REAL ddfv = fprimeprime(*x, arg);

        REAL g = 2 * fv * dfv;
        REAL h = (2 * dfv * dfv - fv * ddfv);
        if (h == 0) return -1;
        delta = -g / h;

        (*x) += delta;
        i += 1;
        if (i > MAXITER) return -2;
    } while (fabs(delta) > TOLERANCE);
    if (SIGN((*x)) != SIGN(x0)) return -3;
    return 0;
}


inline int newton(
    REAL x0,
    REAL *x,
    REAL *arg)
{
    int i = 0;
    REAL delta;

    *x = x0;
    do {
        REAL fv = f(*x, arg);
        REAL dfv = fprime(*x, arg);

        REAL g = fv;
        REAL h = dfv;
        if (h == 0) return -1;
        delta = -g / h;

        (*x) += delta;
        i += 1;
        if (i > MAXITER) return -2;
    } while (fabs(delta) > TOLERANCE);
    if (SIGN((*x)) != SIGN(x0)) return -3;
    return 0;
}


inline void set_new_pos_vel(
    const REAL dt,
    const REAL s,
    const REAL r0,
    const REAL m,
    const REAL e2,
    const REAL alpha,
    REAL *rx,
    REAL *ry,
    REAL *rz,
    REAL *vx,
    REAL *vy,
    REAL *vz)
{
    REAL r0x = *rx;
    REAL r0y = *ry;
    REAL r0z = *rz;
    REAL v0x = *vx;
    REAL v0y = *vy;
    REAL v0z = *vz;

    REAL lf = lagrange_f(s, r0, m, alpha);
    REAL lg = lagrange_g(dt, s, m, alpha);
    REAL r1x, r1y, r1z;
    r1x = r0x * lf + v0x * lg;
    r1y = r0y * lf + v0y * lg;
    r1z = r0z * lf + v0z * lg;

    REAL r1sqr = r1x * r1x + r1y * r1y + r1z * r1z;
    REAL r1 = sqrt(r1sqr);   // XXX: +e2

    REAL ldf = lagrange_dfds(s, r0, r1, m, alpha);
    REAL ldg = lagrange_dgds(s, r1, m, alpha);
    REAL v1x, v1y, v1z;
    v1x = r0x * ldf + v0x * ldg;
    v1y = r0y * ldf + v0y * ldg;
    v1z = r0z * ldf + v0z * ldg;

    *rx = r1x;
    *ry = r1y;
    *rz = r1z;
    *vx = v1x;
    *vy = v1y;
    *vz = v1z;
}


inline void universal_kepler_solver(
    const REAL dt,
    const REAL m,
    const REAL e2,
    const REAL r0x,
    const REAL r0y,
    const REAL r0z,
    const REAL v0x,
    const REAL v0y,
    const REAL v0z,
    REAL *r1x,
    REAL *r1y,
    REAL *r1z,
    REAL *v1x,
    REAL *v1y,
    REAL *v1z)
{
    int i;
    int err = 0;
    int nsteps = 1;
    REAL rx = r0x;
    REAL ry = r0y;
    REAL rz = r0z;
    REAL vx = v0x;
    REAL vy = v0y;
    REAL vz = v0z;

    do {
        err = 0;
        REAL dt0 = dt / nsteps;
        for (i = 0; i < nsteps; ++i) {
            REAL r0sqr = rx * rx + ry * ry + rz * rz;
            if (r0sqr > 0) {
                REAL r0 = sqrt(r0sqr);   // XXX: +e2
                REAL r0v0 = rx * vx + ry * vy + rz * vz;
                REAL v0sqr = vx * vx + vy * vy + vz * vz;
                REAL alpha = 2 * m / r0 - v0sqr;

                REAL s0, s, arg[5];

                s0 = dt0 / r0;
                arg[0] = dt0;
                arg[1] = r0;
                arg[2] = r0v0;
                arg[3] = m;
                arg[4] = alpha;

//                err = newton(s0, &s, arg);
//                err = halley(s0, &s, arg);
                err = laguerre(s0, &s, arg);
                if (err == 0) {
                    set_new_pos_vel(dt0, s, r0, m, e2, alpha,
                                    &rx, &ry, &rz, &vx, &vy, &vz);
                } else {
                    rx = r0x;
                    ry = r0y;
                    rz = r0z;
                    vx = v0x;
                    vy = v0y;
                    vz = v0z;
                    nsteps *= 2;
                    i = nsteps;    // break
                }
            }
        }
    } while (err != 0);

    *r1x = rx;
    *r1y = ry;
    *r1z = rz;
    *v1x = vx;
    *v1y = vy;
    *v1z = vz;
}

