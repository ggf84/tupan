#ifndef __SMOOTHING_H__
#define __SMOOTHING_H__

#include "common.h"

void smoothed_inv_r1(
    REALn r2,
    REALn h2,
    REALn *inv_r1);

void smoothed_inv_r2(
    REALn r2,
    REALn h2,
    REALn *inv_r2);

void smoothed_inv_r3(
    REALn r2,
    REALn h2,
    REALn *inv_r3);

void smoothed_inv_r1r2(
    REALn r2,
    REALn h2,
    REALn *inv_r1,
    REALn *inv_r2);

void smoothed_inv_r1r3(
    REALn r2,
    REALn h2,
    REALn *inv_r1,
    REALn *inv_r3);

void smoothed_inv_r2r3(
    REALn r2,
    REALn h2,
    REALn *inv_r2,
    REALn *inv_r3);

void smoothed_inv_r1r2r3(
    REALn r2,
    REALn h2,
    REALn *inv_r1,
    REALn *inv_r2,
    REALn *inv_r3);

#endif  // __SMOOTHING_H__
