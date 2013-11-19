#include "acc_kernel_common.h"


__kernel void acc_kernel(
    const UINT ni,
    __global const REAL * restrict _im,
    __global const REAL * restrict _irx,
    __global const REAL * restrict _iry,
    __global const REAL * restrict _irz,
    __global const REAL * restrict _ie2,
    const UINT nj,
    __global const REAL * restrict _jm,
    __global const REAL * restrict _jrx,
    __global const REAL * restrict _jry,
    __global const REAL * restrict _jrz,
    __global const REAL * restrict _je2,
    __global REAL * restrict _iax,
    __global REAL * restrict _iay,
    __global REAL * restrict _iaz)
{
    UINT gid = get_global_id(0);
    gid = min(WPT * VW * gid, (ni - WPT * VW));

    REALn im[WPT], irx[WPT], iry[WPT], irz[WPT], ie2[WPT];

    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) im[i] = vloadn(i, _im + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) irx[i] = vloadn(i, _irx + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) iry[i] = vloadn(i, _iry + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) irz[i] = vloadn(i, _irz + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) ie2[i] = vloadn(i, _ie2 + gid);

    REALn iax[WPT], iay[WPT], iaz[WPT];

    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) iax[i] = (REALn)(0);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) iay[i] = (REALn)(0);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) iaz[i] = (REALn)(0);

#ifdef FAST_LOCAL_MEM
    __local REAL __jm[LSIZE];
    __local REAL __jrx[LSIZE];
    __local REAL __jry[LSIZE];
    __local REAL __jrz[LSIZE];
    __local REAL __je2[LSIZE];
    UINT j = 0;
    UINT stride = min((UINT)(get_local_size(0)), (UINT)(LSIZE));
    for (; stride > 0; stride /= 2) {
        UINT lid = get_local_id(0) % stride;
        for (; (j + stride - 1) < nj; j += stride) {
            barrier(CLK_LOCAL_MEM_FENCE);
            __jm[lid] = _jm[j + lid];
            __jrx[lid] = _jrx[j + lid];
            __jry[lid] = _jry[j + lid];
            __jrz[lid] = _jrz[j + lid];
            __je2[lid] = _je2[j + lid];
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll UNROLL
            for (UINT k = 0; k < stride; ++k) {
                #pragma unroll
                for (UINT i = 0; i < WPT; ++i) {
                    acc_kernel_core(im[i], irx[i], iry[i], irz[i], ie2[i],
                                    __jm[k], __jrx[k], __jry[k], __jrz[k], __je2[k],
                                    &iax[i], &iay[i], &iaz[i]);
                }
            }
        }
    }
#else
    #pragma unroll UNROLL
    for (UINT j = 0; j < nj; ++j) {
        #pragma unroll
        for (UINT i = 0; i < WPT; ++i) {
            acc_kernel_core(im[i], irx[i], iry[i], irz[i], ie2[i],
                            _jm[j], _jrx[j], _jry[j], _jrz[j], _je2[j],
                            &iax[i], &iay[i], &iaz[i]);
        }
    }
#endif

    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) vstoren(iax[i], i, _iax + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) vstoren(iay[i], i, _iay + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) vstoren(iaz[i], i, _iaz + gid);
}

