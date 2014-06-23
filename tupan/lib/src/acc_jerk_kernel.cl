#include "acc_jerk_kernel_common.h"


__kernel
__attribute__((reqd_work_group_size(LSIZE, 1, 1)))
void acc_jerk_kernel(
    const UINT ni,
    __global const REALn * restrict _im,
    __global const REALn * restrict _irx,
    __global const REALn * restrict _iry,
    __global const REALn * restrict _irz,
    __global const REALn * restrict _ie2,
    __global const REALn * restrict _ivx,
    __global const REALn * restrict _ivy,
    __global const REALn * restrict _ivz,
    const UINT nj,
    __global const REAL * restrict _jm,
    __global const REAL * restrict _jrx,
    __global const REAL * restrict _jry,
    __global const REAL * restrict _jrz,
    __global const REAL * restrict _je2,
    __global const REAL * restrict _jvx,
    __global const REAL * restrict _jvy,
    __global const REAL * restrict _jvz,
    __global REALn * restrict _iax,
    __global REALn * restrict _iay,
    __global REALn * restrict _iaz,
    __global REALn * restrict _ijx,
    __global REALn * restrict _ijy,
    __global REALn * restrict _ijz)
{
    for (UINT i = LSIZE * get_group_id(0);
         VW * i < ni; i += LSIZE * get_num_groups(0)) {
        UINT lid = get_local_id(0);
        UINT gid = i + lid;
        gid = ((VW * gid) < ni) ? (gid):(0);

        REALn im = _im[gid];
        REALn irx = _irx[gid];
        REALn iry = _iry[gid];
        REALn irz = _irz[gid];
        REALn ie2 = _ie2[gid];
        REALn ivx = _ivx[gid];
        REALn ivy = _ivy[gid];
        REALn ivz = _ivz[gid];

        REALn iax = (REALn)(0);
        REALn iay = (REALn)(0);
        REALn iaz = (REALn)(0);
        REALn ijx = (REALn)(0);
        REALn ijy = (REALn)(0);
        REALn ijz = (REALn)(0);

        UINT j = 0;

        #ifdef FAST_LOCAL_MEM
        for (; (j + LSIZE - 1) < nj; j += LSIZE) {
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
                acc_jerk_kernel_core(
                    im, irx, iry, irz,
                    ie2, ivx, ivy, ivz,
                    __jm[k], __jrx[k], __jry[k], __jrz[k],
                    __je2[k], __jvx[k], __jvy[k], __jvz[k],
                    &iax, &iay, &iaz,
                    &ijx, &ijy, &ijz);
            }
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

        _iax[gid] = iax;
        _iay[gid] = iay;
        _iaz[gid] = iaz;
        _ijx[gid] = ijx;
        _ijy[gid] = ijy;
        _ijz[gid] = ijz;
    }
}

