#ifndef SMOOTHING_H
#define SMOOTHING_H

#include"common.h"


////////////////////////////////////////////////////////////////////////////////
// phi smoothing

inline REAL
phi_plummer_smooth(REAL r2, REAL h2)
{
    REAL inv_r2 = 1 / (r2 + h2);
    inv_r2 = (r2 > 0) ? (inv_r2):(0);
    REAL inv_r = sqrt(inv_r2);
    return inv_r;
}

inline REAL
phi_smooth(REAL r2, REAL h2)
{
    return phi_plummer_smooth(r2, h2);
}


////////////////////////////////////////////////////////////////////////////////
// acc smoothing

inline REAL
acc_plummer_smooth(REAL r2, REAL h2)
{
    REAL inv_r2 = 1 / (r2 + h2);
    inv_r2 = (r2 > 0) ? (inv_r2):(0);
    REAL inv_r = sqrt(inv_r2);
    REAL inv_r3 = inv_r * inv_r2;
    return inv_r3;
}

inline REAL
acc_smooth(REAL r2, REAL h2)
{
    return acc_plummer_smooth(r2, h2);
}


////////////////////////////////////////////////////////////////////////////////
// acc-jerk smoothing

inline REAL2
accjerk_plummer_smooth(REAL r2, REAL h2)
{
    REAL inv_r2 = 1 / (r2 + h2);
    inv_r2 = (r2 > 0) ? (inv_r2):(0);
    REAL inv_r = sqrt(inv_r2);
    REAL inv_r3 = inv_r * inv_r2;
    return (REAL2){inv_r2, inv_r3};
}

inline REAL2
accjerk_smooth(REAL r2, REAL h2)
{
    return accjerk_plummer_smooth(r2, h2);
}


////////////////////////////////////////////////////////////////////////////////
// rho smoothing

inline REAL
rho_plummer_smooth(REAL r2, REAL h2)
{
    REAL inv_r2 = 1 / (r2 + h2);
    inv_r2 = (r2 > 0) ? (inv_r2):(0);
    REAL inv_r = sqrt(inv_r2);
    REAL inv_r5 = inv_r * inv_r2 * inv_r2;
    REAL h2_r5 = THREE_FOURPI * h2 * inv_r5;
    return h2_r5;
}

inline REAL
rho_smooth(REAL r2, REAL h2)
{
    return rho_plummer_smooth(r2, h2);
}

#endif  // SMOOTHING_H

