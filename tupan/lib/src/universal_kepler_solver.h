#ifndef __UNIVERSAL_KEPLER_SOLVER_H__
#define __UNIVERSAL_KEPLER_SOLVER_H__

#include "common.h"

void
universal_kepler_solver(
    const REAL dt0,
    const REAL4 pos0,
    const REAL4 vel0,
    REAL4 *pos1,
    REAL4 *vel1);

#endif  // __UNIVERSAL_KEPLER_SOLVER_H__
