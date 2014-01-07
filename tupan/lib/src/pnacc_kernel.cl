#include "pnacc_kernel_common.h"


__kernel void pnacc_kernel(
    const UINT ni,
    __global const REAL * restrict _im,
    __global const REAL * restrict _irx,
    __global const REAL * restrict _iry,
    __global const REAL * restrict _irz,
    __global const REAL * restrict _ie2,
    __global const REAL * restrict _ivx,
    __global const REAL * restrict _ivy,
    __global const REAL * restrict _ivz,
    const UINT nj,
    __global const REAL * restrict _jm,
    __global const REAL * restrict _jrx,
    __global const REAL * restrict _jry,
    __global const REAL * restrict _jrz,
    __global const REAL * restrict _je2,
    __global const REAL * restrict _jvx,
    __global const REAL * restrict _jvy,
    __global const REAL * restrict _jvz,
    const UINT order,
    const REAL inv1,
    const REAL inv2,
    const REAL inv3,
    const REAL inv4,
    const REAL inv5,
    const REAL inv6,
    const REAL inv7,
    __global REAL * restrict _ipnax,
    __global REAL * restrict _ipnay,
    __global REAL * restrict _ipnaz)
{
    UINT gid = get_global_id(0) * WPT * VW;

    UINT imask[WPT];

    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        imask[i] = (VW * i + gid) < ni;

    CLIGHT clight = CLIGHT_Init(order, inv1, inv2, inv3, inv4, inv5, inv6, inv7);

    REALn im[WPT], irx[WPT], iry[WPT], irz[WPT],
          ie2[WPT], ivx[WPT], ivy[WPT], ivz[WPT];

    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            im[i] = vloadn(i, _im + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            irx[i] = vloadn(i, _irx + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            iry[i] = vloadn(i, _iry + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            irz[i] = vloadn(i, _irz + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            ie2[i] = vloadn(i, _ie2 + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            ivx[i] = vloadn(i, _ivx + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            ivy[i] = vloadn(i, _ivy + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            ivz[i] = vloadn(i, _ivz + gid);

    REALn ipnax[WPT], ipnay[WPT], ipnaz[WPT];

    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            ipnax[i] = (REALn)(0);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            ipnay[i] = (REALn)(0);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            ipnaz[i] = (REALn)(0);

#ifdef FAST_LOCAL_MEM
    __local REAL __jm[LSIZE];
    __local REAL __jrx[LSIZE];
    __local REAL __jry[LSIZE];
    __local REAL __jrz[LSIZE];
    __local REAL __je2[LSIZE];
    __local REAL __jvx[LSIZE];
    __local REAL __jvy[LSIZE];
    __local REAL __jvz[LSIZE];
    UINT j = 0;
    UINT lid = get_local_id(0);
    for (UINT stride = get_local_size(0); stride > 0; stride /= 2) {
        INT mask = lid < stride;
        for (; (j + stride - 1) < nj; j += stride) {
            if (mask) {
                __jm[lid] = _jm[j + lid];
                __jrx[lid] = _jrx[j + lid];
                __jry[lid] = _jry[j + lid];
                __jrz[lid] = _jrz[j + lid];
                __je2[lid] = _je2[j + lid];
                __jvx[lid] = _jvx[j + lid];
                __jvy[lid] = _jvy[j + lid];
                __jvz[lid] = _jvz[j + lid];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll UNROLL
            for (UINT k = 0; k < stride; ++k) {
                #pragma unroll
                for (UINT i = 0; i < WPT; ++i) {
                    pnacc_kernel_core(im[i], irx[i], iry[i], irz[i],
                                      ie2[i], ivx[i], ivy[i], ivz[i],
                                      __jm[k], __jrx[k], __jry[k], __jrz[k],
                                      __je2[k], __jvx[k], __jvy[k], __jvz[k],
                                      clight,
                                      &ipnax[i], &ipnay[i], &ipnaz[i]);
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
#else
    #pragma unroll UNROLL
    for (UINT j = 0; j < nj; ++j) {
        #pragma unroll
        for (UINT i = 0; i < WPT; ++i) {
            pnacc_kernel_core(im[i], irx[i], iry[i], irz[i],
                              ie2[i], ivx[i], ivy[i], ivz[i],
                              _jm[j], _jrx[j], _jry[j], _jrz[j],
                              _je2[j], _jvx[j], _jvy[j], _jvz[j],
                              clight,
                              &ipnax[i], &ipnay[i], &ipnaz[i]);
        }
    }
#endif

    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            vstoren(ipnax[i], i, _ipnax + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            vstoren(ipnay[i], i, _ipnay + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i)
        if (imask[i])
            vstoren(ipnaz[i], i, _ipnaz + gid);
}

