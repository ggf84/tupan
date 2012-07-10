#ifndef UNIVERSAL_KEPLER_SOLVER_H
#define UNIVERSAL_KEPLER_SOLVER_H

#include"common.h"


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
    return (stumpff_C(zeta) - 3 * stumpff_S(zeta)) / (2*z);
}


inline REAL
stumpff_S_prime(const REAL zeta)
{
    return (1 - stumpff_S(zeta) - 2 * stumpff_C(zeta)) / (2*z);
}


inline REAL
lagrange_f(const REAL xi,
           const REAL r0,
           const REAL vr0,
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
               const REAL vr0,
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
           const REAL vr0,
           const REAL smu,
           const REAL alpha)
{
    REAL zeta = alpha * xi * xi;
    REAL xi_smu = xi / smu;
    REAL r0xi_smu = r0 * xi_smu;
    return r0xi_smu * (xi_smu * vr0 * stumpff_C(zeta) - (zeta * stumpff_S(zeta) - 1));
}


inline REAL
lagrange_dgdxi(const REAL xi,
               const REAL r0,
               const REAL vr0,
               const REAL smu,
               const REAL alpha)
{
    REAL zeta = alpha * xi * xi;
    REAL xi_smu = xi / smu;
    REAL r0_smu = r0 / smu;
    return r0_smu * (vr0 * xi_smu * (1 - zeta * stumpff_S(zeta)) - (zeta * stumpff_C(zeta) - 1));
}


inline REAL
universal_kepler(const REAL xi,
                 const REAL r0,
                 const REAL vr0,
                 const REAL smu,
                 const REAL alpha)
{
    REAL xi2 = xi * xi;
    REAL zeta = alpha * xi2;
    REAL xi_smu = xi / smu;
    return xi * (r0 * (1 + vr0 * xi_smu * stumpff_C(zeta)) + (1 - alpha * r0) * xi2 * stumpff_S(zeta));
}


inline REAL
universal_kepler_dxi(const REAL xi,
                     const REAL r0,
                     const REAL vr0,
                     const REAL smu,
                     const REAL alpha)
{
    REAL xi2 = xi * xi;
    REAL zeta = alpha * xi2;
    REAL xi_smu = xi / smu;
    return r0 * (1 + vr0 * xi_smu * (1 - zeta * stumpff_S(zeta))) + (1 - alpha * r0) * xi2 * stumpff_C(zeta);
}


inline REAL
universal_kepler_dxidxi(const REAL xi,
                        const REAL r0,
                        const REAL vr0,
                        const REAL smu,
                        const REAL alpha)
{
    return xi + r0 * vr0 / smu - alpha * universal_kepler(xi, r0, vr0, smu, alpha);
}




#endif // UNIVERSAL_KEPLER_SOLVER_H

