#ifndef UNIVERSAL_KEPLER_SOLVER_H
#define UNIVERSAL_KEPLER_SOLVER_H

#include"common.h"


#ifdef DOUBLE
    #define TOLERANCE  1.0536712127723509e-8   // sqrt(2^-53)
#else
    #define TOLERANCE  2.44140625E-4            // sqrt(2^-24)
#endif
#define MAXITER 128
#define SIGN(x) (((x) > 0) - ((x) < 0))


inline REAL
stumpff_c0(const REAL zeta)
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


inline REAL
stumpff_c1(const REAL zeta)
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


inline REAL
stumpff_c2(const REAL zeta)
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


inline REAL
stumpff_c3(const REAL zeta)
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


inline REAL
lagrange_f(const REAL s,
           const REAL r0,
           const REAL mu,
           const REAL alpha)
{
    REAL s2 = s * s;
    REAL zeta = alpha * s2;
    REAL mu_r0 = mu / r0;
    return 1 - mu_r0 * s2 * stumpff_c2(zeta);
}


inline REAL
lagrange_dfds(const REAL s,
              const REAL r0,
              const REAL r1,
              const REAL mu,
              const REAL alpha)
{
    REAL s2 = s * s;
    REAL zeta = alpha * s2;
    REAL mu_r0r1 = mu / (r0 * r1);
    return -mu_r0r1 * s * stumpff_c1(zeta);
}


inline REAL
lagrange_g(const REAL dt,
           const REAL s,
           const REAL mu,
           const REAL alpha)
{
    REAL s2 = s * s;
    REAL zeta = alpha * s2;
    return dt - mu * s * s2 * stumpff_c3(zeta);
}


inline REAL
lagrange_dgds(const REAL s,
              const REAL r1,
              const REAL mu,
              const REAL alpha)
{
    REAL s2 = s * s;
    REAL zeta = alpha * s2;
    REAL mu_r1 = mu / r1;
    return 1 - mu_r1 * s2 * stumpff_c2(zeta);
}


inline REAL
universal_kepler(const REAL s,
                 const REAL r0,
                 const REAL rv0,
                 const REAL mu,
                 const REAL alpha)
{

    REAL s2 = s * s;
    REAL zeta = alpha * s2;
    return r0 * s * stumpff_c1(zeta) + rv0 * s2 * stumpff_c2(zeta) + mu * s * s2 * stumpff_c3(zeta);
}


inline REAL
universal_kepler_ds(const REAL s,
                    const REAL r0,
                    const REAL rv0,
                    const REAL mu,
                    const REAL alpha)
{

    REAL s2 = s * s;
    REAL zeta = alpha * s2;
    return r0 * stumpff_c0(zeta) + rv0 * s * stumpff_c1(zeta) + mu * s2 * stumpff_c2(zeta);
}


inline REAL
universal_kepler_dsds(const REAL s,
                      const REAL r0,
                      const REAL rv0,
                      const REAL mu,
                      const REAL alpha)
{

    REAL s2 = s * s;
    REAL zeta = alpha * s2;
    return (mu - alpha * r0) * s * stumpff_c1(zeta) + rv0 * stumpff_c0(zeta);
}


inline REAL
f(const REAL s,
  REAL *arg)
{
    return universal_kepler(s, arg[1], arg[2], arg[3], arg[4]) - arg[0];
}


inline REAL
fprime(const REAL s,
       REAL *arg)
{
    return universal_kepler_ds(s, arg[1], arg[2], arg[3], arg[4]);
}


inline REAL
fprimeprime(const REAL s,
            REAL *arg)
{
    return universal_kepler_dsds(s, arg[1], arg[2], arg[3], arg[4]);
}


#define ORDER 4
inline int
laguerre(REAL x0,
         REAL *x,
         REAL *arg
        )
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


inline int
halley(REAL x0,
       REAL *x,
       REAL *arg
      )
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


inline int
newton(REAL x0,
       REAL *x,
       REAL *arg
      )
{
    int i = 0;
    REAL delta;

    *x = x0;
    do {
        REAL fv = f(*x, arg);
        REAL dfv = fprime(*x, arg);
/*
        REAL alpha = 0.5;
        if (fv * dfv < 0) {
            alpha = -0.5;
        }
        dfv += alpha * fv;
*/
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


inline void
set_new_pos_vel(const REAL dt,
                const REAL s,
                const REAL r0,
                const REAL mu,
                const REAL alpha,
                REAL4 *pos,
                REAL4 *vel)
{
    REAL4 pos0 = *pos;
    REAL4 vel0 = *vel;

    REAL lf = lagrange_f(s, r0, mu, alpha);
    REAL lg = lagrange_g(dt, s, mu, alpha);
    REAL4 pos1;
    pos1.x = pos0.x * lf + vel0.x * lg;
    pos1.y = pos0.y * lf + vel0.y * lg;
    pos1.z = pos0.z * lf + vel0.z * lg;
    pos1.w = pos0.w;

    REAL r1sqr = pos1.x * pos1.x + pos1.y * pos1.y + pos1.z * pos1.z;
    REAL r1 = sqrt(r1sqr);

    REAL ldf = lagrange_dfds(s, r0, r1, mu, alpha);
    REAL ldg = lagrange_dgds(s, r1, mu, alpha);
    REAL4 vel1;
    vel1.x = pos0.x * ldf + vel0.x * ldg;
    vel1.y = pos0.y * ldf + vel0.y * ldg;
    vel1.z = pos0.z * ldf + vel0.z * ldg;
    vel1.w = vel0.w;

    *pos = pos1;
    *vel = vel1;
}


inline void
universal_kepler_solver(const REAL dt0,
                        const REAL4 pos0,
                        const REAL4 vel0,
                        REAL4 *pos1,
                        REAL4 *vel1)
{
    REAL4 pos = pos0;
    REAL4 vel = vel0;
    int err = -1;
    int counter = 1;

    while (err != 0) {
        REAL dt = dt0 / counter;
        int i;
        for (i = 0; i < counter; ++i) {

            REAL mu = pos.w;
            REAL r0sqr = pos.x * pos.x + pos.y * pos.y + pos.z * pos.z;
            REAL v0sqr = vel.x * vel.x + vel.y * vel.y + vel.z * vel.z;
            REAL rv0 = pos.x * vel.x + pos.y * vel.y + pos.z * vel.z;
            REAL r0 = sqrt(r0sqr);

            REAL alpha = 2 * mu / r0 - v0sqr;

            REAL s0, s, arg[5];

            s0 = dt / r0;
            arg[0] = dt;
            arg[1] = r0;
            arg[2] = rv0;
            arg[3] = mu;
            arg[4] = alpha;

//            err = laguerre(s0, &s, arg);
//            err = halley(s0, &s, arg);
            err = newton(s0, &s, arg);

            if (err == 0) {
                set_new_pos_vel(dt, s, r0, mu, alpha, &pos, &vel);
            } else {
//                printf("ERROR: %d %d\ns: %.17g s0: %.17g\narg: %.17g %.17g %.17g %.17g %.17g\n",
//                       err, count, s, s0, arg[0], arg[1], arg[2], arg[3], arg[4]);
                pos = pos0;
                vel = vel0;
                counter *= 2;
                i = counter;    // break
            }
        }
    }

    *pos1 = pos;
    *vel1 = vel;
}



#endif // UNIVERSAL_KEPLER_SOLVER_H

