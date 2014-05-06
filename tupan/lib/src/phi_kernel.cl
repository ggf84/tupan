#include "phi_kernel_common.h"


__kernel void phi_kernel(
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
    __global REAL * restrict _iphi)
{
    UINT lid = get_local_id(0);
    UINT lsize = get_local_size(0);
    UINT i = VW * lsize * get_group_id(0);

    if ((i + VW * lid) >= ni) {
        i -= VW * lid;
    }

    REALn im = vloadn(lid, _im + i);
    REALn irx = vloadn(lid, _irx + i);
    REALn iry = vloadn(lid, _iry + i);
    REALn irz = vloadn(lid, _irz + i);
    REALn ie2 = vloadn(lid, _ie2 + i);

    REALn iphi = (REALn)(0);

    UINT j = 0;

    #ifdef FAST_LOCAL_MEM
        __local REAL __jm[LSIZE];
        __local REAL __jrx[LSIZE];
        __local REAL __jry[LSIZE];
        __local REAL __jrz[LSIZE];
        __local REAL __je2[LSIZE];
        for (; (j + lsize - 1) < nj; j += lsize) {
            __jm[lid] = _jm[j + lid];
            __jrx[lid] = _jrx[j + lid];
            __jry[lid] = _jry[j + lid];
            __jrz[lid] = _jrz[j + lid];
            __je2[lid] = _je2[j + lid];
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll UNROLL
            for (UINT k = 0; k < lsize; ++k) {
                phi_kernel_core(
                    im, irx, iry, irz, ie2,
                    __jm[k], __jrx[k], __jry[k], __jrz[k], __je2[k],
                    &iphi);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    #endif

    #pragma unroll UNROLL
    for (; j < nj; ++j) {
        phi_kernel_core(
            im, irx, iry, irz, ie2,
            _jm[j], _jrx[j], _jry[j], _jrz[j], _je2[j],
            &iphi);
    }

    vstoren(iphi, lid, _iphi + i);
}

