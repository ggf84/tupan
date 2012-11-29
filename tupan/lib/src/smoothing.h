#ifndef SMOOTHING_H
#define SMOOTHING_H

#include"common.h"


////////////////////////////////////////////////////////////////////////////////
// smoothed inv_r1

inline REAL
plummer_smoothed_inv_r1(REAL r2, REAL h2)
{
    REAL inv_r2 = 1 / (r2 + h2);
    inv_r2 = (r2 > 0) ? (inv_r2):(0);
    REAL inv_r = sqrt(inv_r2);
    return inv_r;
}

inline REAL
smoothed_inv_r1(REAL r2, REAL h2)
{
    return plummer_smoothed_inv_r1(r2, h2);
}


////////////////////////////////////////////////////////////////////////////////
// smoothed inv_r2

inline REAL
plummer_smoothed_inv_r2(REAL r2, REAL h2)
{
    REAL inv_r2 = 1 / (r2 + h2);
    inv_r2 = (r2 > 0) ? (inv_r2):(0);
    return inv_r2;
}

inline REAL
smoothed_inv_r2(REAL r2, REAL h2)
{
    return plummer_smoothed_inv_r2(r2, h2);
}


////////////////////////////////////////////////////////////////////////////////
// smoothed inv_r3

inline REAL
plummer_smoothed_inv_r3(REAL r2, REAL h2)
{
    REAL inv_r2 = 1 / (r2 + h2);
    inv_r2 = (r2 > 0) ? (inv_r2):(0);
    REAL inv_r = sqrt(inv_r2);
    REAL inv_r3 = inv_r * inv_r2;
    return inv_r3;
}

inline REAL
smoothed_inv_r3(REAL r2, REAL h2)
{
    return plummer_smoothed_inv_r3(r2, h2);
}


////////////////////////////////////////////////////////////////////////////////
// smoothed inv_r1r2

inline REAL2
plummer_smoothed_inv_r1r2(REAL r2, REAL h2)
{
    REAL inv_r2 = 1 / (r2 + h2);
    inv_r2 = (r2 > 0) ? (inv_r2):(0);
    REAL inv_r = sqrt(inv_r2);
    return (REAL2){inv_r, inv_r2};
}

inline REAL2
smoothed_inv_r1r2(REAL r2, REAL h2)
{
    return plummer_smoothed_inv_r1r2(r2, h2);
}


////////////////////////////////////////////////////////////////////////////////
// smoothed inv_r1r3

inline REAL2
plummer_smoothed_inv_r1r3(REAL r2, REAL h2)
{
    REAL inv_r2 = 1 / (r2 + h2);
    inv_r2 = (r2 > 0) ? (inv_r2):(0);
    REAL inv_r = sqrt(inv_r2);
    REAL inv_r3 = inv_r * inv_r2;
    return (REAL2){inv_r, inv_r3};
}

inline REAL2
smoothed_inv_r1r3(REAL r2, REAL h2)
{
    return plummer_smoothed_inv_r1r3(r2, h2);
}


////////////////////////////////////////////////////////////////////////////////
// smoothed inv_r2r3

inline REAL2
plummer_smoothed_inv_r2r3(REAL r2, REAL h2)
{
    REAL inv_r2 = 1 / (r2 + h2);
    inv_r2 = (r2 > 0) ? (inv_r2):(0);
    REAL inv_r = sqrt(inv_r2);
    REAL inv_r3 = inv_r * inv_r2;
    return (REAL2){inv_r2, inv_r3};
}

inline REAL2
smoothed_inv_r2r3(REAL r2, REAL h2)
{
    return plummer_smoothed_inv_r2r3(r2, h2);
}


////////////////////////////////////////////////////////////////////////////////
// smoothed inv_r1r2r3

inline REAL3
plummer_smoothed_inv_r1r2r3(REAL r2, REAL h2)
{
    REAL inv_r2 = 1 / (r2 + h2);
    inv_r2 = (r2 > 0) ? (inv_r2):(0);
    REAL inv_r = sqrt(inv_r2);
    REAL inv_r3 = inv_r * inv_r2;
    return (REAL3){inv_r, inv_r2, inv_r3};
}

inline REAL3
smoothed_inv_r1r2r3(REAL r2, REAL h2)
{
    return plummer_smoothed_inv_r1r2r3(r2, h2);
}


#endif  // SMOOTHING_H
