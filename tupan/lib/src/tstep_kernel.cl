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
    UINT lid = get_local_id(0);
    UINT lsize = get_local_size(0);
    UINT gid = get_global_id(0);
    gid *= ((VW * gid) < ni);

    REALn im = vloadn(gid, _im);
    REALn irx = vloadn(gid, _irx);
    REALn iry = vloadn(gid, _iry);
    REALn irz = vloadn(gid, _irz);
    REALn ie2 = vloadn(gid, _ie2);
    REALn ivx = vloadn(gid, _ivx);
    REALn ivy = vloadn(gid, _ivy);
    REALn ivz = vloadn(gid, _ivz);

    REALn iw2_a = (REALn)(0);
    REALn iw2_b = (REALn)(0);

    UINT j = 0;

    #ifdef FAST_LOCAL_MEM
    for (; (j + lsize - 1) < nj; j += lsize) {
        __local REAL __jm[LSIZE];
        __local REAL __jrx[LSIZE];
        __local REAL __jry[LSIZE];
        __local REAL __jrz[LSIZE];
        __local REAL __je2[LSIZE];
        __local REAL __jvx[LSIZE];
        __local REAL __jvy[LSIZE];
        __local REAL __jvz[LSIZE];
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
        for (UINT k = 0; k < lsize; ++k) {
            tstep_kernel_core(
                eta,
                im, irx, iry, irz,
                ie2, ivx, ivy, ivz,
                __jm[k], __jrx[k], __jry[k], __jrz[k],
                __je2[k], __jvx[k], __jvy[k], __jvz[k],
                &iw2_a, &iw2_b);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #endif

    #pragma unroll UNROLL
    for (; j < nj; ++j) {
        tstep_kernel_core(
            eta,
            im, irx, iry, irz,
            ie2, ivx, ivy, ivz,
            _jm[j], _jrx[j], _jry[j], _jrz[j],
            _je2[j], _jvx[j], _jvy[j], _jvz[j],
            &iw2_a, &iw2_b);
    }

    vstoren(eta / sqrt(1 + iw2_a), gid, _idt_a);
    vstoren(eta / sqrt(1 + iw2_b), gid, _idt_b);
}

