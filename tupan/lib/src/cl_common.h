#ifndef __CL_COMMON_H__
#define __CL_COMMON_H__

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

#endif // __CL_COMMON_H__
