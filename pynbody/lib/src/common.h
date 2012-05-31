#ifndef COMMON_H
#define COMMON_H

#ifdef __OPENCL_VERSION__
    #ifdef DOUBLE
        typedef double REAL;
        typedef double2 REAL2;
        typedef double3 REAL3;
        typedef double4 REAL4;
        typedef double8 REAL8;
        typedef double16 REAL16;
    #else
        typedef float REAL;
        typedef float2 REAL2;
        typedef float3 REAL3;
        typedef float4 REAL4;
        typedef float8 REAL8;
        typedef float16 REAL16;
    #endif
#else
    #include<math.h>
    #ifdef DOUBLE
        typedef double REAL;
        #define rsqrt(x) (1.0/sqrt(x))
    #else
        typedef float REAL;
        #define rsqrt(x) (1.0f/sqrtf(x))
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
    typedef struct real8_struct {
        REAL s0;
        REAL s1;
        REAL s2;
        REAL s3;
        REAL s4;
        REAL s5;
        REAL s6;
        REAL s7;
    } REAL8, *pREAL8;
    typedef struct real16_struct {
        REAL s0;
        REAL s1;
        REAL s2;
        REAL s3;
        REAL s4;
        REAL s5;
        REAL s6;
        REAL s7;
        REAL s8;
        REAL s9;
        REAL sa;
        REAL sb;
        REAL sc;
        REAL sd;
        REAL se;
        REAL sf;
    } REAL16, *pREAL16;
#endif // __OPENCL_VERSION__


#define PI ((REAL)3.141592653589793)
#define PI2	((REAL)9.869604401089358)
#define TWOPI ((REAL)6.283185307179586)
#define FOURPI ((REAL)12.566370614359172)
#define THREE_FOURPI ((REAL)0.238732414637843)


typedef struct clight_struct {
    REAL inv1;
    REAL inv2;
    REAL inv3;
    REAL inv4;
    REAL inv5;
    REAL inv6;
    REAL inv7;
    uint order;
} CLIGHT, *pCLIGHT;

#endif // COMMON_H

