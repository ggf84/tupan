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
    UINT lid = get_local_id(0);
    UINT lsize = get_local_size(0);
    UINT i = VW * lsize * get_group_id(0);

    if ((i + VW * lid) >= ni) {
        i -= VW * lid;
    }

    CLIGHT clight = CLIGHT_Init(order, inv1, inv2, inv3, inv4, inv5, inv6, inv7);

    REALn im = vloadn(lid, _im + i);
    REALn irx = vloadn(lid, _irx + i);
    REALn iry = vloadn(lid, _iry + i);
    REALn irz = vloadn(lid, _irz + i);
    REALn ie2 = vloadn(lid, _ie2 + i);
    REALn ivx = vloadn(lid, _ivx + i);
    REALn ivy = vloadn(lid, _ivy + i);
    REALn ivz = vloadn(lid, _ivz + i);

    REALn ipnax = (REALn)(0);
    REALn ipnay = (REALn)(0);
    REALn ipnaz = (REALn)(0);

    UINT j = 0;

    #ifdef FAST_LOCAL_MEM
        __local REAL __jm[LSIZE];
        __local REAL __jrx[LSIZE];
        __local REAL __jry[LSIZE];
        __local REAL __jrz[LSIZE];
        __local REAL __je2[LSIZE];
        __local REAL __jvx[LSIZE];
        __local REAL __jvy[LSIZE];
        __local REAL __jvz[LSIZE];
        for (; (j + lsize - 1) < nj; j += lsize) {
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
                pnacc_kernel_core(
                    im, irx, iry, irz,
                    ie2, ivx, ivy, ivz,
                    __jm[k], __jrx[k], __jry[k], __jrz[k],
                    __je2[k], __jvx[k], __jvy[k], __jvz[k],
                    clight,
                    &ipnax, &ipnay, &ipnaz);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    #endif

    #pragma unroll UNROLL
    for (; j < nj; ++j) {
        pnacc_kernel_core(
            im, irx, iry, irz,
            ie2, ivx, ivy, ivz,
            _jm[j], _jrx[j], _jry[j], _jrz[j],
            _je2[j], _jvx[j], _jvy[j], _jvz[j],
            clight,
            &ipnax, &ipnay, &ipnaz);
    }

    vstoren(ipnax, lid, _ipnax + i);
    vstoren(ipnay, lid, _ipnay + i);
    vstoren(ipnaz, lid, _ipnaz + i);
}

