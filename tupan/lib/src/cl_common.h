#ifndef __CL_COMMON_H__
#define __CL_COMMON_H__

#ifdef CONFIG_USE_DOUBLE
    #if !defined(CL_VERSION_1_2)
        #if defined(cl_khr_fp64)
            #pragma OPENCL EXTENSION cl_khr_fp64 : enable
        #else
            #error "The hardware/OpenCL implementation does not support double precision arithmetic."
        #endif
    #endif
#endif

#ifdef CONFIG_USE_DOUBLE
    typedef long INT;
    typedef long2 INT2;
    typedef long4 INT4;
    typedef long8 INT8;
    typedef long16 INT16;

    typedef ulong UINT;
    typedef ulong2 UINT2;
    typedef ulong4 UINT4;
    typedef ulong8 UINT8;
    typedef ulong16 UINT16;

    typedef double REAL;
    typedef double2 REAL2;
    typedef double4 REAL4;
    typedef double8 REAL8;
    typedef double16 REAL16;
#else
    typedef int INT;
    typedef int2 INT2;
    typedef int4 INT4;
    typedef int8 INT8;
    typedef int16 INT16;

    typedef uint UINT;
    typedef uint2 UINT2;
    typedef uint4 UINT4;
    typedef uint8 UINT8;
    typedef uint16 UINT16;

    typedef float REAL;
    typedef float2 REAL2;
    typedef float4 REAL4;
    typedef float8 REAL8;
    typedef float16 REAL16;
#endif

#define paster(x,y) x##y
#define concat(x,y) paster(x,y)
#define vec(x) concat(x, VW)

#define INT1 INT
#define UINT1 UINT
#define REAL1 REAL

#define INTn vec(INT)
#define UINTn vec(UINT)
#define REALn vec(REAL)

#endif // __CL_COMMON_H__
