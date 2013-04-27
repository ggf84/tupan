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

typedef struct real2_struct {
    REAL x, y;
} REAL2;

typedef struct real3_struct {
    REAL x, y, z;
} REAL3;

typedef struct real4_struct {
    REAL x, y, z, w;
} REAL4;

typedef struct real8_struct {
    REAL s0, s1, s2, s3, s4, s5, s6, s7;
} REAL8;

typedef struct real16_struct {
    REAL s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, sa, sb, sc, sd, se, sf;
} REAL16;

#endif // __C_COMMON_H__
