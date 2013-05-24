#ifndef __UNIVERSAL_KEPLER_SOLVER_H__
#define __UNIVERSAL_KEPLER_SOLVER_H__

#include "common.h"

void universal_kepler_solver(
    const REAL dt,
    const REAL m,
    const REAL e2,
    const REAL r0x,
    const REAL r0y,
    const REAL r0z,
    const REAL v0x,
    const REAL v0y,
    const REAL v0z,
    REAL *r1x,
    REAL *r1y,
    REAL *r1z,
    REAL *v1x,
    REAL *v1y,
    REAL *v1z);

#endif  // __UNIVERSAL_KEPLER_SOLVER_H__
