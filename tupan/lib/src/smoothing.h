#ifndef SMOOTHING_H
#define SMOOTHING_H

#include "common.h"


inline REAL
smoothed_inv_r1(REAL r2, REAL h2);

inline REAL
smoothed_inv_r2(REAL r2, REAL h2);

inline REAL
smoothed_inv_r3(REAL r2, REAL h2);

inline REAL2
smoothed_inv_r1r2(REAL r2, REAL h2);

inline REAL2
smoothed_inv_r1r3(REAL r2, REAL h2);

inline REAL2
smoothed_inv_r2r3(REAL r2, REAL h2);

inline REAL3
smoothed_inv_r1r2r3(REAL r2, REAL h2);

#endif  // !SMOOTHING_H
