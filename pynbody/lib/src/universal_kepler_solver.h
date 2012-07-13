#ifndef UNIVERSAL_KEPLER_SOLVER_H
#define UNIVERSAL_KEPLER_SOLVER_H

#include"common.h"


#define TOLERANCE  1.0536712127723509e-08   // sqrt(2^-53)
//#define TOLERANCE  2.44140625E-4            // sqrt(2^-24)
#define MAXITER 100
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



typedef REAL (ftype)(const REAL x, REAL *arg);

#define ORDER 8
inline int
laguerre(REAL x0,
         REAL *x,
         REAL *arg,
         REAL tol,
         ftype *f,
         ftype *fprime,
         ftype *fprimeprime)
{
    int i = 0;
    REAL delta;

    *x = x0;
    do {
        REAL fv = (*f)(*x, arg);
        REAL dfv = (*fprime)(*x, arg);
        REAL ddfv = (*fprimeprime)(*x, arg);
        if (fv == 0 || dfv == 0  || ddfv == 0) return 0;
        REAL g = dfv / fv;
        REAL g2 = g * g;
        REAL h = g2 - ddfv / fv;
        delta = -ORDER / (g + SIGN(g) * sqrt(fabs((ORDER - 1) * (ORDER * h - g2))));
        (*x) += delta;
        i += 1;
        if (i > MAXITER) return -1;
//        printf("%d %e %e\n", i, delta, tol);
    } while (fabs(delta * (*x + x0)) > 2 * tol * fabs(*x - x0));
    if (SIGN((*x)) != SIGN(x0)) return -3;
    return 0;
}


inline REAL8
universal_kepler_solver(const REAL dt0,
                        const REAL4 pos0,
                        const REAL4 vel0)
{
    REAL mu = pos0.w;
    REAL r0sqr = pos0.x * pos0.x + pos0.y * pos0.y + pos0.z * pos0.z;
    REAL v0sqr = vel0.x * vel0.x + vel0.y * vel0.y + vel0.z * vel0.z;
    REAL rv0 = pos0.x * vel0.x + pos0.y * vel0.y + pos0.z * vel0.z;
    REAL r0 = sqrt(r0sqr);

    REAL alpha = 2 * mu / r0 - v0sqr;

    REAL dt = dt0;
/*
    REAL P = 2 * PI * sqrt(alpha * alpha * alpha) / (mu * mu);
    double nP;
    if (dt0 > P) {
        dt = modf(dt0/P, &nP) * P;
//        printf("%e %e %e\n", dt0, P, dt);
    }
*/
    REAL s0 = dt / r0;

    REAL s, arg[5], tol;

    arg[0] = dt;
    arg[1] = r0;
    arg[2] = rv0;
    arg[3] = mu;
    arg[4] = alpha;
    tol = TOLERANCE;

    int err = laguerre(s0, &s, arg, tol, &f, &fprime, &fprimeprime);
    if (err != 0) {
        fprintf(stderr, "ERROR: %d\ns: %.17g s0: %.17g\narg: %.17g %.17g %.17g %.17g %.17g\n",
                err, s, s0, arg[0], arg[1], arg[2], arg[3], arg[4]);
    }

    REAL lf = lagrange_f(s, r0, mu, alpha);
    REAL lg = lagrange_g(dt, s, mu, alpha);
    REAL4 pos1;
    pos1.x = pos0.x * lf + vel0.x * lg;
    pos1.y = pos0.y * lf + vel0.y * lg;
    pos1.z = pos0.z * lf + vel0.z * lg;
    pos1.w = 0;

    REAL r1sqr = pos1.x * pos1.x + pos1.y * pos1.y + pos1.z * pos1.z;
    REAL r1 = sqrt(r1sqr);

    REAL ldf = lagrange_dfds(s, r0, r1, mu, alpha);
    REAL ldg = lagrange_dgds(s, r1, mu, alpha);
    REAL4 vel1;
    vel1.x = pos0.x * ldf + vel0.x * ldg;
    vel1.y = pos0.y * ldf + vel0.y * ldg;
    vel1.z = pos0.z * ldf + vel0.z * ldg;
    vel1.w = 0;

    REAL8 posvel = {pos1.x, pos1.y, pos1.z, pos1.w, vel1.x, vel1.y, vel1.z, vel1.w};
    return posvel;
}



#endif // UNIVERSAL_KEPLER_SOLVER_H

