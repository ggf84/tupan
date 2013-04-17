#ifndef UNIVERSAL_KEPLER_SOLVER_H
#define UNIVERSAL_KEPLER_SOLVER_H

#include "common.h"


inline void
universal_kepler_solver(const REAL dt0,
                        const REAL4 pos0,
                        const REAL4 vel0,
                        REAL4 *pos1,
                        REAL4 *vel1);

#endif  // !UNIVERSAL_KEPLER_SOLVER_H
