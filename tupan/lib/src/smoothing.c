#include "common.h"
#include "smoothing.h"

//
// smoothed inv_r1
//
inline void plummer_smoothed_inv_r1(
    REALn r2,
    REALn h2,
    REALn *inv_r1)
{
    INTn mask = (r2 > 0);
    REALn inv_r2 = 1 / (r2 + h2);
    inv_r2 = select((REALn)(0), inv_r2, mask);
    *inv_r1 = sqrt(inv_r2);
}
// Total flop count: 3

inline void smoothed_inv_r1(
    REALn r2,
    REALn h2,
    REALn *inv_r1)
{
    plummer_smoothed_inv_r1(r2, h2, &(*inv_r1));
}

//
// smoothed inv_r2
//
inline void plummer_smoothed_inv_r2(
    REALn r2,
    REALn h2,
    REALn *inv_r2)
{
    INTn mask = (r2 > 0);
    *inv_r2 = 1 / (r2 + h2);
    *inv_r2 = select((REALn)(0), *inv_r2, mask);
}
// Total flop count: 2

inline void smoothed_inv_r2(
    REALn r2,
    REALn h2,
    REALn *inv_r2)
{
    plummer_smoothed_inv_r2(r2, h2, &(*inv_r2));
}

//
// smoothed inv_r3
//
inline void plummer_smoothed_inv_r3(
    REALn r2,
    REALn h2,
    REALn *inv_r3)
{
    INTn mask = (r2 > 0);
    REALn inv_r2 = 1 / (r2 + h2);
    inv_r2 = select((REALn)(0), inv_r2, mask);
    REALn inv_r1 = sqrt(inv_r2);
    *inv_r3 = inv_r1 * inv_r2;
}
// Total flop count: 4

inline void smoothed_inv_r3(
    REALn r2,
    REALn h2,
    REALn *inv_r3)
{
    plummer_smoothed_inv_r3(r2, h2, &(*inv_r3));
}

//
// smoothed inv_r1r2
//
inline void plummer_smoothed_inv_r1r2(
    REALn r2,
    REALn h2,
    REALn *inv_r1,
    REALn *inv_r2)
{
    INTn mask = (r2 > 0);
    *inv_r2 = 1 / (r2 + h2);
    *inv_r2 = select((REALn)(0), *inv_r2, mask);
    *inv_r1 = sqrt(*inv_r2);
}
// Total flop count: 3

inline void smoothed_inv_r1r2(
    REALn r2,
    REALn h2,
    REALn *inv_r1,
    REALn *inv_r2)
{
    plummer_smoothed_inv_r1r2(r2, h2, &(*inv_r1), &(*inv_r2));
}

//
// smoothed inv_r1r3
//
inline void plummer_smoothed_inv_r1r3(
    REALn r2,
    REALn h2,
    REALn *inv_r1,
    REALn *inv_r3)
{
    INTn mask = (r2 > 0);
    REALn inv_r2 = 1 / (r2 + h2);
    inv_r2 = select((REALn)(0), inv_r2, mask);
    *inv_r1 = sqrt(inv_r2);
    *inv_r3 = *inv_r1 * inv_r2;
}
// Total flop count: 4

inline void smoothed_inv_r1r3(
    REALn r2,
    REALn h2,
    REALn *inv_r1,
    REALn *inv_r3)
{
    plummer_smoothed_inv_r1r3(r2, h2, &(*inv_r1), &(*inv_r3));
}

//
// smoothed inv_r2r3
//
inline void plummer_smoothed_inv_r2r3(
    REALn r2,
    REALn h2,
    REALn *inv_r2,
    REALn *inv_r3)
{
    INTn mask = (r2 > 0);
    *inv_r2 = 1 / (r2 + h2);
    *inv_r2 = select((REALn)(0), *inv_r2, mask);
    REALn inv_r1 = sqrt(*inv_r2);
    *inv_r3 = inv_r1 * *inv_r2;
}
// Total flop count: 4

inline void smoothed_inv_r2r3(
    REALn r2,
    REALn h2,
    REALn *inv_r2,
    REALn *inv_r3)
{
    plummer_smoothed_inv_r2r3(r2, h2, &(*inv_r2), &(*inv_r3));
}

//
// smoothed inv_r1r2r3
//
inline void plummer_smoothed_inv_r1r2r3(
    REALn r2,
    REALn h2,
    REALn *inv_r1,
    REALn *inv_r2,
    REALn *inv_r3)
{
    INTn mask = (r2 > 0);
    *inv_r2 = 1 / (r2 + h2);
    *inv_r2 = select((REALn)(0), *inv_r2, mask);
    *inv_r1 = sqrt(*inv_r2);
    *inv_r3 = *inv_r1 * *inv_r2;
}
// Total flop count: 4

inline void smoothed_inv_r1r2r3(
    REALn r2,
    REALn h2,
    REALn *inv_r1,
    REALn *inv_r2,
    REALn *inv_r3)
{
    plummer_smoothed_inv_r1r2r3(r2, h2, &(*inv_r1), &(*inv_r2), &(*inv_r3));
}

