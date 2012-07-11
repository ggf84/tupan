#ifndef UNIVERSAL_KEPLER_SOLVER_H
#define UNIVERSAL_KEPLER_SOLVER_H

#include"common.h"


#define TOLERANCE  5.9604644775390625E-8
#define ORDER  4
#define MAXITER 50
#define SIGN(x)   (((x) > 0) - ((x) < 0))


inline REAL
stumpff_C(const REAL zeta)
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
stumpff_S(const REAL zeta)
{
    if (zeta > 0) {
        REAL sz = sqrt(zeta);
        return (sz - sin(sz)) / (sz * zeta);
    }
    if (zeta < 0) {
        REAL sz = sqrt(-zeta);
        return (sz - sinh(sz)) / (sz * zeta);
    }
    return 1/((REAL)6);
}


inline REAL
stumpff_C_prime(const REAL zeta)
{
    return (stumpff_C(zeta) - 3 * stumpff_S(zeta)) / (2*zeta);
}


inline REAL
stumpff_S_prime(const REAL zeta)
{
    return (1 - stumpff_S(zeta) - 2 * stumpff_C(zeta)) / (2*zeta);
}


inline REAL
lagrange_f(const REAL xi,
           const REAL r0,
           const REAL rv0,
           const REAL smu,
           const REAL alpha)
{
    REAL xi2 = xi * xi;
    REAL xi2_r0 = xi2 / r0;
    return 1 - xi2_r0 * stumpff_C(alpha * xi2);
}


inline REAL
lagrange_dfdxi(const REAL xi,
               const REAL r0,
               const REAL rv0,
               const REAL smu,
               const REAL alpha)
{
    REAL zeta = alpha * xi * xi;
    REAL xi_r0 = xi / r0;
    return xi_r0 * (zeta * stumpff_S(zeta) - 1);
}


inline REAL
lagrange_g(const REAL xi,
           const REAL r0,
           const REAL rv0,
           const REAL smu,
           const REAL alpha)
{
    REAL zeta = alpha * xi * xi;
    REAL xi_smu = xi / smu;
    REAL r0xi_smu = r0 * xi_smu;
    return r0xi_smu * (xi_smu * rv0 * stumpff_C(zeta) - (zeta * stumpff_S(zeta) - 1));
}


inline REAL
lagrange_dgdxi(const REAL xi,
               const REAL r0,
               const REAL rv0,
               const REAL smu,
               const REAL alpha)
{
    REAL zeta = alpha * xi * xi;
    REAL xi_smu = xi / smu;
    REAL r0_smu = r0 / smu;
    return r0_smu * (rv0 * xi_smu * (1 - zeta * stumpff_S(zeta)) - (zeta * stumpff_C(zeta) - 1));
}


inline REAL
universal_kepler(const REAL xi,
                 const REAL r0,
                 const REAL rv0,
                 const REAL smu,
                 const REAL alpha)
{
    REAL xi2 = xi * xi;
    REAL zeta = alpha * xi2;
    REAL xi_smu = xi / smu;
    return xi * (r0 * (1 + rv0 * xi_smu * stumpff_C(zeta)) + (1 - alpha * r0) * xi2 * stumpff_S(zeta));
}


inline REAL
universal_kepler_dxi(const REAL xi,
                     const REAL r0,
                     const REAL rv0,
                     const REAL smu,
                     const REAL alpha)
{
    REAL xi2 = xi * xi;
    REAL zeta = alpha * xi2;
    REAL xi_smu = xi / smu;
    return r0 * (1 + rv0 * xi_smu * (1 - zeta * stumpff_S(zeta))) + (1 - alpha * r0) * xi2 * stumpff_C(zeta);
}


inline REAL
universal_kepler_dxidxi(const REAL xi,
                        const REAL r0,
                        const REAL rv0,
                        const REAL smu,
                        const REAL alpha)
{
    return xi + r0 * rv0 / smu - alpha * universal_kepler(xi, r0, rv0, smu, alpha);
}


inline REAL
f(const REAL xi,
  REAL *arg)
{
    return universal_kepler(xi, arg[1], arg[2], arg[3], arg[4]) - arg[0];
}


inline REAL
fprime(const REAL xi,
       REAL *arg)
{
    return universal_kepler_dxi(xi, arg[1], arg[2], arg[3], arg[4]);
}


inline REAL
fprimeprime(const REAL xi,
            REAL *arg)
{
    return universal_kepler_dxidxi(xi, arg[1], arg[2], arg[3], arg[4]);
}



typedef REAL (ftype)(const REAL x, REAL *arg);

inline int
laguerre(REAL x0,
         REAL *x,
         REAL *arg,
         REAL xtol,
         REAL ytol,
         ftype *f,
         ftype *fprime,
         ftype *fprimeprime)
{
    int i = 0;
    REAL fv, dfv, ddfv, delta;

    *x=x0;
    delta = 2*xtol;
    while (fabs(delta) > xtol) {
        fv = (*f)(*x, arg);
        if (fabs(fv) <= ytol) return 0;
        dfv = (*fprime)(*x, arg);
        ddfv = (*fprimeprime)(*x, arg);
        if (dfv == 0  || ddfv == 0) return -2;
        delta = -ORDER * fv / (dfv + SIGN(dfv) * sqrt(fabs((ORDER - 1) * (ORDER - 1) * dfv * dfv - ORDER * (ORDER - 1) * fv * ddfv)));
        (*x) += delta;
//        printf("%d\n", i);
        i += 1;
        if (i > MAXITER) return -1;
    }
    return 0;
}


inline REAL8
universal_kepler_solver(const REAL dt,
                        const REAL4 pos0,
                        const REAL4 vel0)
{
    REAL mu = pos0.w;
    REAL smu = sqrt(mu);
    REAL r0sqr = pos0.x * pos0.x + pos0.y * pos0.y + pos0.z * pos0.z;
    REAL v0sqr = vel0.x * vel0.x + vel0.y * vel0.y + vel0.z * vel0.z;
    REAL rv0 = pos0.x * vel0.x + pos0.y * vel0.y + pos0.z * vel0.z;
    REAL r0 = sqrt(r0sqr);
    rv0 /= r0;

    REAL alpha = 2 / r0 - v0sqr / mu;

    REAL xi0 = smu * dt / r0;

    REAL xi, arg[5], xtol, ytol;

    arg[0] = smu * dt;
    arg[1] = r0;
    arg[2] = rv0;
    arg[3] = smu;
    arg[4] = alpha;
    ytol = fabs(TOLERANCE * smu * dt);
    xtol = fabs(TOLERANCE * xi0);

//    err = findroot(xi0, &xi, arg, xtol, ytol, &f, &fprime, &fprimeprime);
    int err = laguerre(xi0, &xi, arg, xtol, ytol, &f, &fprime, &fprimeprime);
    if (err != 0 || SIGN(xi) != SIGN(dt)) {
        printf("ERROR:\nxi: %.17g xi0: %.17g\narg: %.17g %.17g %.17g %.17g %.17g\n",
                xi, xi0, arg[0], arg[1], arg[2], arg[3], arg[4]);

        exit(0);
    }

    REAL lf = lagrange_f(xi, r0, rv0, smu, alpha);
    REAL lg = lagrange_g(xi, r0, rv0, smu, alpha);
    REAL ldf = lagrange_dfdxi(xi, r0, rv0, smu, alpha);
    REAL ldg = lagrange_dgdxi(xi, r0, rv0, smu, alpha);

    REAL4 pos1;
    pos1.x = pos0.x * lf + vel0.x * lg;
    pos1.y = pos0.y * lf + vel0.y * lg;
    pos1.z = pos0.z * lf + vel0.z * lg;
    pos1.w = 0;

    REAL r1sqr = pos1.x * pos1.x + pos1.y * pos1.y + pos1.z * pos1.z;
    REAL r1 = sqrt(r1sqr);
    REAL smu_r = smu / r1;

    REAL4 vel1;
    vel1.x = (pos0.x * ldf + vel0.x * ldg) * smu_r;
    vel1.y = (pos0.y * ldf + vel0.y * ldg) * smu_r;
    vel1.z = (pos0.z * ldf + vel0.z * ldg) * smu_r;
    vel1.w = 0;

    REAL8 posvel = {pos1.x, pos1.y, pos1.z, pos1.w, vel1.x, vel1.y, vel1.z, vel1.w};
    return posvel;
}



#endif // UNIVERSAL_KEPLER_SOLVER_H

