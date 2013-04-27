#include "common.h"

//
// smoothed inv_r1
//
inline void
plummer_smoothed_inv_r1(
    REAL r2,
    REAL h2,
    REAL *inv_r1)
{
    int mask1 = (r2 > 0);
    int mask0 = 1^mask1;

    REAL inv_r2 = mask1 / (r2 + h2 + mask0);
    *inv_r1 = sqrt(inv_r2);
}
// Total flop count: 4

inline void
smoothed_inv_r1(
    REAL r2,
    REAL h2,
    REAL *inv_r1)
{
    plummer_smoothed_inv_r1(r2, h2, &(*inv_r1));
}

//
// smoothed inv_r2
//
inline void
plummer_smoothed_inv_r2(
    REAL r2,
    REAL h2,
    REAL *inv_r2)
{
    int mask1 = (r2 > 0);
    int mask0 = 1^mask1;

    *inv_r2 = mask1 / (r2 + h2 + mask0);
}
// Total flop count: 3

inline void
smoothed_inv_r2(
    REAL r2,
    REAL h2,
    REAL *inv_r2)
{
    plummer_smoothed_inv_r2(r2, h2, &(*inv_r2));
}

//
// smoothed inv_r3
//
inline void
plummer_smoothed_inv_r3(
    REAL r2,
    REAL h2,
    REAL *inv_r3)
{
    int mask1 = (r2 > 0);
    int mask0 = 1^mask1;

    REAL inv_r2 = mask1 / (r2 + h2 + mask0);
    REAL inv_r1 = sqrt(inv_r2);
    *inv_r3 = inv_r1 * inv_r2;
}
// Total flop count: 5

inline void
smoothed_inv_r3(
    REAL r2,
    REAL h2,
    REAL *inv_r3)
{
    plummer_smoothed_inv_r3(r2, h2, &(*inv_r3));
}

//
// smoothed inv_r1r2
//
inline void
plummer_smoothed_inv_r1r2(
    REAL r2,
    REAL h2,
    REAL *inv_r1,
    REAL *inv_r2)
{
    int mask1 = (r2 > 0);
    int mask0 = 1^mask1;

    *inv_r2 = mask1 / (r2 + h2 + mask0);
    *inv_r1 = sqrt(*inv_r2);
}
// Total flop count: 4

inline void
smoothed_inv_r1r2(
    REAL r2,
    REAL h2,
    REAL *inv_r1,
    REAL *inv_r2)
{
    plummer_smoothed_inv_r1r2(r2, h2, &(*inv_r1), &(*inv_r2));
}

//
// smoothed inv_r1r3
//
inline void
plummer_smoothed_inv_r1r3(
    REAL r2,
    REAL h2,
    REAL *inv_r1,
    REAL *inv_r3)
{
    int mask1 = (r2 > 0);
    int mask0 = 1^mask1;

    REAL inv_r2 = mask1 / (r2 + h2 + mask0);
    *inv_r1 = sqrt(inv_r2);
    *inv_r3 = *inv_r1 * inv_r2;
}
// Total flop count: 5

inline void
smoothed_inv_r1r3(
    REAL r2,
    REAL h2,
    REAL *inv_r1,
    REAL *inv_r3)
{
    plummer_smoothed_inv_r1r3(r2, h2, &(*inv_r1), &(*inv_r3));
}

//
// smoothed inv_r2r3
//
inline void
plummer_smoothed_inv_r2r3(
    REAL r2,
    REAL h2,
    REAL *inv_r2,
    REAL *inv_r3)
{
    int mask1 = (r2 > 0);
    int mask0 = 1^mask1;

    *inv_r2 = mask1 / (r2 + h2 + mask0);
    REAL inv_r1 = sqrt(*inv_r2);
    *inv_r3 = inv_r1 * *inv_r2;
}
// Total flop count: 5

inline void
smoothed_inv_r2r3(
    REAL r2,
    REAL h2,
    REAL *inv_r2,
    REAL *inv_r3)
{
    plummer_smoothed_inv_r2r3(r2, h2, &(*inv_r2), &(*inv_r3));
}

//
// smoothed inv_r1r2r3
//
inline void
plummer_smoothed_inv_r1r2r3(
    REAL r2,
    REAL h2,
    REAL *inv_r1,
    REAL *inv_r2,
    REAL *inv_r3)
{
    int mask1 = (r2 > 0);
    int mask0 = 1^mask1;

    *inv_r2 = mask1 / (r2 + h2 + mask0);
    *inv_r1 = sqrt(*inv_r2);
    *inv_r3 = *inv_r1 * *inv_r2;
}
// Total flop count: 5

inline void
smoothed_inv_r1r2r3(
    REAL r2,
    REAL h2,
    REAL *inv_r1,
    REAL *inv_r2,
    REAL *inv_r3)
{
    plummer_smoothed_inv_r1r2r3(r2, h2, &(*inv_r1), &(*inv_r2), &(*inv_r3));
}

