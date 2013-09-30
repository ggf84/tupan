#ifndef __COMMON_H__
#define __COMMON_H__

#ifdef __OPENCL_VERSION__
    #include "cl_common.h"
#else
    #include "c_common.h"
#endif

#define PI ((REAL)(3.141592653589793))
#define PI2 ((REAL)(9.869604401089358))
#define TWOPI ((REAL)(6.283185307179586))
#define FOURPI ((REAL)(12.566370614359172))
#define THREE_FOURPI ((REAL)(0.238732414637843))

typedef struct pn_terms {
    REAL a, b;
} PN, *pPN;

typedef struct clight_struct {
    REAL inv1;
    REAL inv2;
    REAL inv3;
    REAL inv4;
    REAL inv5;
    REAL inv6;
    REAL inv7;
    UINT order;
} CLIGHT, *pCLIGHT;

#endif // __COMMON_H__
