#ifndef __COMMON_H__
#define __COMMON_H__

#ifdef CONFIG_USE_OPENCL
    #include "cl_common.h"
#else
    #include "c_common.h"
#endif

#define PI ((real_t)(3.14159265358979323846))
#define PI2 ((real_t)(9.86960440108935861883))
#define TWOPI ((real_t)(6.28318530717958647693))
#define FOURPI ((real_t)(1.25663706143591729539e+1))
#define THREE_FOURPI ((real_t)(2.3873241463784300365e-1))

#ifdef CONFIG_USE_DOUBLE
    #define TOLERANCE exp2((real_t)(-42))
#else
    #define TOLERANCE exp2((real_t)(-16))
#endif
#define MAXITER 64


typedef struct clight_struct {
    real_t inv1;
    real_t inv2;
    real_t inv3;
    real_t inv4;
    real_t inv5;
    real_t inv6;
    real_t inv7;
    uint_t order;
} CLIGHT;
#define CLIGHT_Init(ORDER, INV1, INV2, INV3, INV4, INV5, INV6, INV7)    \
    (CLIGHT){.order=(uint_t)(ORDER), .inv1=(real_t)(INV1),              \
             .inv2=(real_t)(INV2), .inv3=(real_t)(INV3),                \
             .inv4=(real_t)(INV4), .inv5=(real_t)(INV5),                \
             .inv6=(real_t)(INV6), .inv7=(real_t)(INV7)}

typedef struct pn_terms {
    real_tn a;
    real_tn b;
} PN;
#define PN_Init(A, B) (PN){.a=(real_tn)(A), .b=(real_tn)(B)}

#endif // __COMMON_H__
