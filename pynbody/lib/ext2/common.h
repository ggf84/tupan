#ifndef COMMON_H
#define COMMON_H

#define PI 3.141592653589793
#define TWOPI 6.283185307179586
#define FOURPI 12.566370614359172
#define THREE_FOURPI 0.238732414637843

#ifdef OPENCL_EXTENSION
    #ifdef DOUBLE
    typedef double REAL;
    typedef double2 REAL2;
    typedef double4 REAL4;
    #else
    typedef float REAL;
    typedef float2 REAL2;
    typedef float4 REAL4;
    #endif
#else
    #include<math.h>
    #ifdef DOUBLE
    typedef double REAL;
    #define rsqrt(x) (1.0/sqrt(x))
//    #define rsqrt(x) (sqrt(1.0/(x)))
    #else
    typedef float REAL;
    #define rsqrt(x) (1.0/sqrtf(x))
//    #define rsqrt(x) (sqrt(1.0/(x)))
    #endif
    #define min(a, b) ({ __typeof__ (a) _a = (a); \
                         __typeof__ (b) _b = (b); \
                         _a < _b ? _a : _b; })
    #define max(a, b) ({ __typeof__ (a) _a = (a); \
                         __typeof__ (b) _b = (b); \
                         _a > _b ? _a : _b; })
    typedef struct real2_struct {
        REAL x;
        REAL y;
    } REAL2, *pREAL2;
    typedef struct real3_struct {
        REAL x;
        REAL y;
        REAL z;
    } REAL3, *pREAL3;
    typedef struct real4_struct {
        REAL x;
        REAL y;
        REAL z;
        REAL w;
    } REAL4, *pREAL4;
    typedef struct clight_struct {
        REAL inv1;
        REAL inv2;
        REAL inv3;
        REAL inv4;
        REAL inv5;
        REAL inv6;
        REAL inv7;
        unsigned int order;
    } CLIGHT, *pCLIGHT;
#endif // OPENCL_EXTENSION

#endif // COMMON_H

