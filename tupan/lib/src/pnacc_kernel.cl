#include "pnacc_kernel_common.h"


__kernel
__attribute__((reqd_work_group_size(LSIZE, 1, 1)))
void pnacc_kernel(
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
    for (UINT i = 0; VW * i < ni; i += get_global_size(0)) {
        UINT gid = i + get_global_id(0);
        gid = ((VW * gid) < ni) ? (gid):(0);

        CLIGHT clight = CLIGHT_Init(order, inv1, inv2, inv3,
                                    inv4, inv5, inv6, inv7);

        REALn im = vloadn(gid, _im);
        REALn irx = vloadn(gid, _irx);
        REALn iry = vloadn(gid, _iry);
        REALn irz = vloadn(gid, _irz);
        REALn ie2 = vloadn(gid, _ie2);
        REALn ivx = vloadn(gid, _ivx);
        REALn ivy = vloadn(gid, _ivy);
        REALn ivz = vloadn(gid, _ivz);

        REALn ipnax = (REALn)(0);
        REALn ipnay = (REALn)(0);
        REALn ipnaz = (REALn)(0);

        UINT j = 0;

        #ifdef FAST_LOCAL_MEM
        for (; (j + LSIZE - 1) < nj; j += LSIZE) {
            UINT lid = get_local_id(0);
            barrier(CLK_LOCAL_MEM_FENCE);
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
            for (UINT k = 0; k < LSIZE; ++k) {
                pnacc_kernel_core(
                    im, irx, iry, irz,
                    ie2, ivx, ivy, ivz,
                    __jm[k], __jrx[k], __jry[k], __jrz[k],
                    __je2[k], __jvx[k], __jvy[k], __jvz[k],
                    clight,
                    &ipnax, &ipnay, &ipnaz);
            }
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

        vstoren(ipnax, gid, _ipnax);
        vstoren(ipnay, gid, _ipnay);
        vstoren(ipnaz, gid, _ipnaz);
    }
}

