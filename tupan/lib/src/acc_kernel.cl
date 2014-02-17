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
    gid = min(VW * gid, ni - VW);

    concat(REAL, VW) im = concat(vload, VW)(0, _im + gid);
    concat(REAL, VW) irx = concat(vload, VW)(0, _irx + gid);
    concat(REAL, VW) iry = concat(vload, VW)(0, _iry + gid);
    concat(REAL, VW) irz = concat(vload, VW)(0, _irz + gid);
    concat(REAL, VW) ie2 = concat(vload, VW)(0, _ie2 + gid);

    concat(REAL, VW) iax = (concat(REAL, VW))(0);
    concat(REAL, VW) iay = (concat(REAL, VW))(0);
    concat(REAL, VW) iaz = (concat(REAL, VW))(0);

    UINT j = 0;

    #ifdef FAST_LOCAL_MEM
        __local REAL __jm[LSIZE];
        __local REAL __jrx[LSIZE];
        __local REAL __jry[LSIZE];
        __local REAL __jrz[LSIZE];
        __local REAL __je2[LSIZE];
        UINT lid = get_local_id(0);
        UINT lsize = get_local_size(0);
        for (; (j + lsize - 1) < nj; j += lsize) {
            __jm[lid] = _jm[j + lid];
            __jrx[lid] = _jrx[j + lid];
            __jry[lid] = _jry[j + lid];
            __jrz[lid] = _jrz[j + lid];
            __je2[lid] = _je2[j + lid];
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll UNROLL
            for (UINT k = 0; k < lsize; ++k) {
                call(acc_kernel_core, VW)(
                    im, irx, iry, irz, ie2,
                    __jm[k], __jrx[k], __jry[k], __jrz[k], __je2[k],
                    &iax, &iay, &iaz);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    #endif

    #pragma unroll UNROLL
    for (; j < nj; ++j) {
        call(acc_kernel_core, VW)(
            im, irx, iry, irz, ie2,
            _jm[j], _jrx[j], _jry[j], _jrz[j], _je2[j],
            &iax, &iay, &iaz);
    }

    concat(vstore, VW)(iax, 0, _iax + gid);
    concat(vstore, VW)(iay, 0, _iay + gid);
    concat(vstore, VW)(iaz, 0, _iaz + gid);
}

