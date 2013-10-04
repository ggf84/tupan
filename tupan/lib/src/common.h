#ifndef __COMMON_H__
#define __COMMON_H__

#ifdef __OPENCL_VERSION__
    #include "cl_common.h"
#else
    #include "c_common.h"
#endif

#define PI ((REALn)(3.141592653589793))
#define PI2 ((REALn)(9.869604401089358))
#define TWOPI ((REALn)(6.283185307179586))
#define FOURPI ((REALn)(12.566370614359172))
#define THREE_FOURPI ((REALn)(0.238732414637843))

typedef struct pn_terms {
    REALn a;
    REALn b;
} PN, *pPN;
#define PN_Init(A, B) (PN){.a=(REALn)(A), .b=(REALn)(B)}

typedef struct clight_struct {
    REALn inv1;
    REALn inv2;
    REALn inv3;
    REALn inv4;
    REALn inv5;
    REALn inv6;
    REALn inv7;
    UINT order;
} CLIGHT, *pCLIGHT;
#define CLIGHT_Init(ORDER, INV1, INV2, INV3, INV4, INV5, INV6, INV7)    \
    (CLIGHT){.order=(UINT)(ORDER), .inv1=(REALn)(INV1), \
             .inv2=(REALn)(INV2), .inv3=(REALn)(INV3),  \
             .inv4=(REALn)(INV4), .inv5=(REALn)(INV5),  \
             .inv6=(REALn)(INV6), .inv7=(REALn)(INV7)}

#endif // __COMMON_H__
