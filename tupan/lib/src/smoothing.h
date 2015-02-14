#ifndef __SMOOTHING_H__
#define __SMOOTHING_H__

#include "common.h"


static inline REALn
get_inv_r(
    REALn const r2,
    REALn const e2,
    INTn const mask)
{
    REALn inv_r = rsqrt(r2 + e2);
    return select((REALn)(0), inv_r, mask);
}
// Total flop count: 2


static inline REALn
smoothed_m_r1(
    REALn const m,
    REALn const r2,
    REALn const e2,
    INTn const mask)
{
    REALn inv_r = get_inv_r(r2, e2, mask);
    return m * inv_r;
}
// Total flop count: 3


static inline REALn
smoothed_m_r2(
    REALn const m,
    REALn const r2,
    REALn const e2,
    INTn const mask)
{
    REALn inv_r = get_inv_r(r2, e2, mask);
    return m * inv_r * inv_r;
}
// Total flop count: 4


static inline REALn
smoothed_m_r3(
    REALn const m,
    REALn const r2,
    REALn const e2,
    INTn const mask)
{
    REALn inv_r = get_inv_r(r2, e2, mask);
    return m * inv_r * inv_r * inv_r;
}
// Total flop count: 5


static inline REALn
smoothed_m_r1_inv_r2(
    REALn const m,
    REALn const r2,
    REALn const e2,
    INTn const mask,
    REALn *inv_r2)
{
    REALn inv_r = get_inv_r(r2, e2, mask);
    *inv_r2 = inv_r * inv_r;
    return m * inv_r;
}
// Total flop count: 4


static inline REALn
smoothed_m_r3_inv_r2(
    REALn const m,
    REALn const r2,
    REALn const e2,
    INTn const mask,
    REALn *inv_r2)
{
    REALn inv_r = get_inv_r(r2, e2, mask);
    *inv_r2 = inv_r * inv_r;
    return m * inv_r * *inv_r2;
}
// Total flop count: 5


static inline REALn
smoothed_m_r1_m_r3(
    REALn const m,
    REALn const r2,
    REALn const e2,
    INTn const mask,
    REALn *m_r3)
{
    REALn inv_r = get_inv_r(r2, e2, mask);
    REALn m_r1 = m * inv_r;
    *m_r3 = inv_r * inv_r * m_r1;
    return m_r1;
}
// Total flop count: 5


static inline REALn
smoothed_inv_r1r2(
    REALn const r2,
    REALn const e2,
    INTn const mask,
    REALn *inv_r2)
{
    REALn inv_r = get_inv_r(r2, e2, mask);
    *inv_r2 = inv_r * inv_r;
    return inv_r;
}
// Total flop count: 3


#endif  // __SMOOTHING_H__
