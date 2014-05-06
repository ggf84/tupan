#ifndef __SMOOTHING_H__
#define __SMOOTHING_H__

#include "common.h"


static inline REALn plummer_smoothed_m_r1(
    REALn m,
    REALn r2,
    REALn e2,
    INTn mask)
{
    REALn inv_r2 = (REALn)(1) / (r2 + e2);
    inv_r2 = select((REALn)(0), inv_r2, mask);
    return m * sqrt(inv_r2);
}
// Total flop count: 4


static inline REALn plummer_smoothed_m_r2(
    REALn m,
    REALn r2,
    REALn e2,
    INTn mask)
{
    REALn inv_r2 = (REALn)(1) / (r2 + e2);
    inv_r2 = select((REALn)(0), inv_r2, mask);
    return m * inv_r2;
}
// Total flop count: 3


static inline REALn plummer_smoothed_m_r3(
    REALn m,
    REALn r2,
    REALn e2,
    INTn mask)
{
    REALn inv_r2 = (REALn)(1) / (r2 + e2);
    inv_r2 = select((REALn)(0), inv_r2, mask);
    return m * inv_r2 * sqrt(inv_r2);
}
// Total flop count: 5


static inline REALn plummer_smoothed_m_r1_inv_r2(
    REALn m,
    REALn r2,
    REALn e2,
    INTn mask,
    REALn *inv_r2)
{
    *inv_r2 = (REALn)(1) / (r2 + e2);
    *inv_r2 = select((REALn)(0), *inv_r2, mask);
    return m * sqrt(*inv_r2);
}
// Total flop count: 4


static inline REALn plummer_smoothed_m_r3_inv_r2(
    REALn m,
    REALn r2,
    REALn e2,
    INTn mask,
    REALn *inv_r2)
{
    *inv_r2 = (REALn)(1) / (r2 + e2);
    *inv_r2 = select((REALn)(0), *inv_r2, mask);
    return m * *inv_r2 * sqrt(*inv_r2);
}
// Total flop count: 5


static inline REALn plummer_smoothed_m_r3_m_r1(
    REALn m,
    REALn r2,
    REALn e2,
    INTn mask,
    REALn *m_r1)
{
    REALn inv_r2 = (REALn)(1) / (r2 + e2);
    inv_r2 = select((REALn)(0), inv_r2, mask);
    *m_r1 = m * sqrt(inv_r2);
    return inv_r2 * *m_r1;
}
// Total flop count: 5


static inline REALn plummer_smoothed_inv_r3r2r1(
    REALn r2,
    REALn h2,
    INTn mask,
    REALn *inv_r2,
    REALn *inv_r1)
{
    *inv_r2 = (REALn)(1) / (r2 + h2);
    *inv_r2 = select((REALn)(0), *inv_r2, mask);
    *inv_r1 = sqrt(*inv_r2);
    return *inv_r2 * *inv_r1;
}
// Total flop count: 4


#define smoothed_m_r1           plummer_smoothed_m_r1
#define smoothed_m_r2           plummer_smoothed_m_r2
#define smoothed_m_r3           plummer_smoothed_m_r3
#define smoothed_m_r1_inv_r2    plummer_smoothed_m_r1_inv_r2
#define smoothed_m_r3_inv_r2    plummer_smoothed_m_r3_inv_r2
#define smoothed_m_r3_m_r1      plummer_smoothed_m_r3_m_r1
#define smoothed_inv_r3r2r1     plummer_smoothed_inv_r3r2r1


#endif  // __SMOOTHING_H__
