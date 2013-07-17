#ifndef __CL_COMMON_H__
#define __CL_COMMON_H__

#if __OPENCL_VERSION__ <= CL_VERSION_1_1
    #ifdef cl_khr_fp64
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #endif
#endif

#ifdef DOUBLE
    typedef double REAL;
#else
    typedef float REAL;
#endif

#endif // __CL_COMMON_H__
