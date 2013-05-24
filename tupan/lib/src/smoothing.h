#ifndef __SMOOTHING_H__
#define __SMOOTHING_H__

#include "common.h"

void smoothed_inv_r1(
    REAL r2,
    REAL h2,
    REAL *inv_r1);

void smoothed_inv_r2(
    REAL r2,
    REAL h2,
    REAL *inv_r2);

void smoothed_inv_r3(
    REAL r2,
    REAL h2,
    REAL *inv_r3);

void smoothed_inv_r1r2(
    REAL r2,
    REAL h2,
    REAL *inv_r1,
    REAL *inv_r2);

void smoothed_inv_r1r3(
    REAL r2,
    REAL h2,
    REAL *inv_r1,
    REAL *inv_r3);

void smoothed_inv_r2r3(
    REAL r2,
    REAL h2,
    REAL *inv_r2,
    REAL *inv_r3);

void smoothed_inv_r1r2r3(
    REAL r2,
    REAL h2,
    REAL *inv_r1,
    REAL *inv_r2,
    REAL *inv_r3);

#endif  // __SMOOTHING_H__
