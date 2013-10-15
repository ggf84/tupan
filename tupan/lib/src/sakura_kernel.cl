#include "sakura_kernel_common.h"


__kernel void sakura_kernel(
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
    const REAL dt,
    const INT flag,
    __global REAL * restrict _idrx,
    __global REAL * restrict _idry,
    __global REAL * restrict _idrz,
    __global REAL * restrict _idvx,
    __global REAL * restrict _idvy,
    __global REAL * restrict _idvz)
{
    UINT i = get_global_id(0);

    REAL im = vload1(i, _im);
    REAL irx = vload1(i, _irx);
    REAL iry = vload1(i, _iry);
    REAL irz = vload1(i, _irz);
    REAL ie2 = vload1(i, _ie2);
    REAL ivx = vload1(i, _ivx);
    REAL ivy = vload1(i, _ivy);
    REAL ivz = vload1(i, _ivz);

    REAL idrx = (REAL)(0);
    REAL idry = (REAL)(0);
    REAL idrz = (REAL)(0);
    REAL idvx = (REAL)(0);
    REAL idvy = (REAL)(0);
    REAL idvz = (REAL)(0);

    UINT j = 0;
    __local REAL __jm[LSIZE];
    __local REAL __jrx[LSIZE];
    __local REAL __jry[LSIZE];
    __local REAL __jrz[LSIZE];
    __local REAL __je2[LSIZE];
    __local REAL __jvx[LSIZE];
    __local REAL __jvy[LSIZE];
    __local REAL __jvz[LSIZE];
    for (; (j + LSIZE) < nj; j += LSIZE) {
        event_t e[8];
        e[0] = async_work_group_copy(__jm, _jm + j, LSIZE, 0);
        e[1] = async_work_group_copy(__jrx, _jrx + j, LSIZE, 0);
        e[2] = async_work_group_copy(__jry, _jry + j, LSIZE, 0);
        e[3] = async_work_group_copy(__jrz, _jrz + j, LSIZE, 0);
        e[4] = async_work_group_copy(__je2, _je2 + j, LSIZE, 0);
        e[5] = async_work_group_copy(__jvx, _jvx + j, LSIZE, 0);
        e[6] = async_work_group_copy(__jvy, _jvy + j, LSIZE, 0);
        e[7] = async_work_group_copy(__jvz, _jvz + j, LSIZE, 0);
        wait_group_events(8, e);
        for (UINT k = 0; k < LSIZE; ++k) {
            sakura_kernel_core(dt, flag,
                               im, irx, iry, irz, ie2, ivx, ivy, ivz,
                               __jm[k], __jrx[k], __jry[k], __jrz[k],
                               __je2[k], __jvx[k], __jvy[k], __jvz[k],
                               &idrx, &idry, &idrz,
                               &idvx, &idvy, &idvz);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (; j < nj; ++j) {
        sakura_kernel_core(dt, flag,
                           im, irx, iry, irz, ie2, ivx, ivy, ivz,
                           _jm[j], _jrx[j], _jry[j], _jrz[j],
                           _je2[j], _jvx[j], _jvy[j], _jvz[j],
                           &idrx, &idry, &idrz,
                           &idvx, &idvy, &idvz);
    }

    vstore1(idrx, i, _idrx);
    vstore1(idry, i, _idry);
    vstore1(idrz, i, _idrz);
    vstore1(idvx, i, _idvx);
    vstore1(idvy, i, _idvy);
    vstore1(idvz, i, _idvz);
}

