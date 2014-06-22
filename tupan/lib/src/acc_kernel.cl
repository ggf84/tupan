#include "acc_kernel_common.h"


__kernel
__attribute__((reqd_work_group_size(LSIZE, 1, 1)))
void acc_kernel(
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
    for (UINT i = LSIZE * get_group_id(0);
         VW * i < ni; i += LSIZE * get_num_groups(0)) {
        UINT lid = get_local_id(0);
        UINT gid = i + lid;
        gid = ((VW * gid) < ni) ? (gid):(0);

        REALn im = vloadn(gid, _im);
        REALn irx = vloadn(gid, _irx);
        REALn iry = vloadn(gid, _iry);
        REALn irz = vloadn(gid, _irz);
        REALn ie2 = vloadn(gid, _ie2);

        REALn iax = (REALn)(0);
        REALn iay = (REALn)(0);
        REALn iaz = (REALn)(0);

        UINT j = 0;

        #ifdef FAST_LOCAL_MEM
        for (; (j + LSIZE - 1) < nj; j += LSIZE) {
            barrier(CLK_LOCAL_MEM_FENCE);
            __local REAL __jm[LSIZE];
            __local REAL __jrx[LSIZE];
            __local REAL __jry[LSIZE];
            __local REAL __jrz[LSIZE];
            __local REAL __je2[LSIZE];
            __jm[lid] = _jm[j + lid];
            __jrx[lid] = _jrx[j + lid];
            __jry[lid] = _jry[j + lid];
            __jrz[lid] = _jrz[j + lid];
            __je2[lid] = _je2[j + lid];
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll UNROLL
            for (UINT k = 0; k < LSIZE; ++k) {
                acc_kernel_core(
                    im, irx, iry, irz, ie2,
                    __jm[k], __jrx[k], __jry[k], __jrz[k], __je2[k],
                    &iax, &iay, &iaz);
            }
        }
        #endif

        #pragma unroll UNROLL
        for (; j < nj; ++j) {
            acc_kernel_core(
                im, irx, iry, irz, ie2,
                _jm[j], _jrx[j], _jry[j], _jrz[j], _je2[j],
                &iax, &iay, &iaz);
        }

        vstoren(iax, gid, _iax);
        vstoren(iay, gid, _iay);
        vstoren(iaz, gid, _iaz);
    }
}

