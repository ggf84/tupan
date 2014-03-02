#include "acc_jerk_kernel_common.h"


__kernel void acc_jerk_kernel(
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
    __global REAL * restrict _iax,
    __global REAL * restrict _iay,
    __global REAL * restrict _iaz,
    __global REAL * restrict _ijx,
    __global REAL * restrict _ijy,
    __global REAL * restrict _ijz)
{
    UINT gid = get_global_id(0);
    gid = min(VW * gid, ni - VW);

    REALn im = vloadn(0, _im + gid);
    REALn irx = vloadn(0, _irx + gid);
    REALn iry = vloadn(0, _iry + gid);
    REALn irz = vloadn(0, _irz + gid);
    REALn ie2 = vloadn(0, _ie2 + gid);
    REALn ivx = vloadn(0, _ivx + gid);
    REALn ivy = vloadn(0, _ivy + gid);
    REALn ivz = vloadn(0, _ivz + gid);

    REALn iax = (REALn)(0);
    REALn iay = (REALn)(0);
    REALn iaz = (REALn)(0);
    REALn ijx = (REALn)(0);
    REALn ijy = (REALn)(0);
    REALn ijz = (REALn)(0);

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
        UINT lid = get_local_id(0);
        UINT lsize = get_local_size(0);
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
                acc_jerk_kernel_core(
                    im, irx, iry, irz,
                    ie2, ivx, ivy, ivz,
                    __jm[k], __jrx[k], __jry[k], __jrz[k],
                    __je2[k], __jvx[k], __jvy[k], __jvz[k],
                    &iax, &iay, &iaz,
                    &ijx, &ijy, &ijz);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    #endif

    #pragma unroll UNROLL
    for (; j < nj; ++j) {
        acc_jerk_kernel_core(
            im, irx, iry, irz,
            ie2, ivx, ivy, ivz,
            _jm[j], _jrx[j], _jry[j], _jrz[j],
            _je2[j], _jvx[j], _jvy[j], _jvz[j],
            &iax, &iay, &iaz,
            &ijx, &ijy, &ijz);
    }

    vstoren(iax, 0, _iax + gid);
    vstoren(iay, 0, _iay + gid);
    vstoren(iaz, 0, _iaz + gid);
    vstoren(ijx, 0, _ijx + gid);
    vstoren(ijy, 0, _ijy + gid);
    vstoren(ijz, 0, _ijz + gid);
}

