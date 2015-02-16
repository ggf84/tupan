#ifndef __CL_COMMON_H__
#define __CL_COMMON_H__

#if !defined(CL_VERSION_1_2)
    #if defined(cl_khr_fp64)
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #else
        #error "Missing double precision extension"
    #endif
#endif

#define DEFINE_TYPE(TYPEA, TYPEB)   \
    typedef TYPEA TYPEB;            \
    typedef TYPEA TYPEB##1;         \
    typedef TYPEA##2 TYPEB##2;      \
    typedef TYPEA##4 TYPEB##4;      \
    typedef TYPEA##8 TYPEB##8;      \
    typedef TYPEA##16 TYPEB##16;

#ifdef CONFIG_USE_DOUBLE
    DEFINE_TYPE(long, int_t)
    DEFINE_TYPE(ulong, uint_t)
    DEFINE_TYPE(double, real_t)
#else
    DEFINE_TYPE(int, int_t)
    DEFINE_TYPE(uint, uint_t)
    DEFINE_TYPE(float, real_t)
#endif

#define paster(x,y) x##y
#define concat(x,y) paster(x,y)
#define vec(x) concat(x, VW)

#define int_tn vec(int_t)
#define uint_tn vec(uint_t)
#define real_tn vec(real_t)

#endif // __CL_COMMON_H__
