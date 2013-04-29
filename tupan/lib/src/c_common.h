#ifndef __C_COMMON_H__
#define __C_COMMON_H__

#include <stdio.h>
#include <math.h>

#define min(a, b) ({ __typeof__ (a) _a = (a); \
                     __typeof__ (b) _b = (b); \
                     _a < _b ? _a : _b; })

#define max(a, b) ({ __typeof__ (a) _a = (a); \
                     __typeof__ (b) _b = (b); \
                     _a > _b ? _a : _b; })

#ifdef DOUBLE
    typedef double REAL;
#else
    typedef float REAL;
#endif

#endif // __C_COMMON_H__
