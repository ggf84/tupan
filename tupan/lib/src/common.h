#ifndef __COMMON_H__
#define __COMMON_H__

#ifdef CONFIG_USE_OPENCL
    #include "cl_common.h"
#else
    #include "c_common.h"
#endif

#define PI ((REAL)(3.141592653589793))
#define PI2 ((REAL)(9.869604401089358))
#define TWOPI ((REAL)(6.283185307179586))
#define FOURPI ((REAL)(12.566370614359172))
#define THREE_FOURPI ((REAL)(0.238732414637843))

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
#define CLIGHT_Init(ORDER, INV1, INV2, INV3, INV4, INV5, INV6, INV7)    \
    (CLIGHT){.order=(UINT)(ORDER), .inv1=(REAL)(INV1), \
             .inv2=(REAL)(INV2), .inv3=(REAL)(INV3),  \
             .inv4=(REAL)(INV4), .inv5=(REAL)(INV5),  \
             .inv6=(REAL)(INV6), .inv7=(REAL)(INV7)}

typedef struct pn_terms {
    REALn a;
    REALn b;
} PN, *pPN;
#define PN_Init(A, B) (PN){.a=(REALn)(A), .b=(REALn)(B)}

#endif // __COMMON_H__
