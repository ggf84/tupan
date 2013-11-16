#include "tstep_kernel_common.h"


__kernel void tstep_kernel(
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
    const REAL eta,
    __global REAL * restrict _idt_a,
    __global REAL * restrict _idt_b)
{
    UINT gid = get_global_id(0);
    gid = min(WPT * VW * gid, (ni - WPT * VW));

    REALn im[WPT], irx[WPT], iry[WPT], irz[WPT],
          ie2[WPT], ivx[WPT], ivy[WPT], ivz[WPT];

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
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) ivx[i] = vloadn(i, _ivx + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) ivy[i] = vloadn(i, _ivy + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) ivz[i] = vloadn(i, _ivz + gid);

    REALn iw2_a[WPT], iw2_b[WPT];

    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) iw2_a[i] = (REALn)(0);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) iw2_b[i] = (REALn)(0);

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
    UINT stride = min((UINT)(get_local_size(0)), (UINT)(LSIZE));
    #pragma unroll 4
    for (; stride > 0; stride /= 2) {
        UINT lid = get_local_id(0) % stride;
        for (; (j + stride - 1) < nj; j += stride) {
            barrier(CLK_LOCAL_MEM_FENCE);
            __jm[lid] = _jm[j + lid];
            __jrx[lid] = _jrx[j + lid];
            __jry[lid] = _jry[j + lid];
            __jrz[lid] = _jrz[j + lid];
            __je2[lid] = _je2[j + lid];
            __jvx[lid] = _jvx[j + lid];
            __jvy[lid] = _jvy[j + lid];
            __jvz[lid] = _jvz[j + lid];
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll UNROLL
            for (UINT k = 0; k < stride; ++k) {
                #pragma unroll
                for (UINT i = 0; i < WPT; ++i) {
                    tstep_kernel_core(eta,
                                      im[i], irx[i], iry[i], irz[i],
                                      ie2[i], ivx[i], ivy[i], ivz[i],
                                      __jm[k], __jrx[k], __jry[k], __jrz[k],
                                      __je2[k], __jvx[k], __jvy[k], __jvz[k],
                                      &iw2_a[i], &iw2_b[i]);
                }
            }
        }
    }
#else
    for (UINT j = 0; j < nj; ++j) {
        #pragma unroll
        for (UINT i = 0; i < WPT; ++i) {
            tstep_kernel_core(eta,
                              im[i], irx[i], iry[i], irz[i],
                              ie2[i], ivx[i], ivy[i], ivz[i],
                              _jm[j], _jrx[j], _jry[j], _jrz[j],
                              _je2[j], _jvx[j], _jvy[j], _jvz[j],
                              &iw2_a[i], &iw2_b[i]);
        }
    }
#endif

    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) vstoren(eta / sqrt(1 + iw2_a[i]), i, _idt_a + gid);
    #pragma unroll
    for (UINT i = 0; i < WPT; ++i) vstoren(eta / sqrt(1 + iw2_b[i]), i, _idt_b + gid);
}

