#ifndef SMOOTHING_H
#define SMOOTHING_H

#include"common.h"


////////////////////////////////////////////////////////////////////////////////
// phi smoothing

inline REAL
phi_plummer_smooth(REAL r2, REAL h2)
{
    REAL rinv = rsqrt(r2 + h2);
    return ((r2 > 0) ? rinv:0);
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
    REAL rinv = rsqrt(r2 + h2);
    REAL rinv3 = rinv * rinv * rinv;
    return ((r2 > 0) ? rinv3:0);
}

inline REAL
acc_smooth(REAL r2, REAL h2)
{
    return acc_plummer_smooth(r2, h2);
}


////////////////////////////////////////////////////////////////////////////////
// rho smoothing

inline REAL
rho_plummer_smooth(REAL r2, REAL h2)
{
    REAL rinv = rsqrt(r2 + h2);
    REAL h2_rinv5 = THREE_FOURPI * h2 * rinv * rinv * rinv * rinv * rinv;
    return ((r2 > 0) ? h2_rinv5:0);
}

inline REAL
rho_smooth(REAL r2, REAL h2)
{
    return rho_plummer_smooth(r2, h2);
}

#endif  // SMOOTHING_H

