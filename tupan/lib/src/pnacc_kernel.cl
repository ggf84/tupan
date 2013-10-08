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
    UINT i = get_global_id(0);

    CLIGHT clight = CLIGHT_Init(order, inv1, inv2, inv3, inv4, inv5, inv6, inv7);

    REALn im = vloadn(i, _im);
    REALn irx = vloadn(i, _irx);
    REALn iry = vloadn(i, _iry);
    REALn irz = vloadn(i, _irz);
    REALn ie2 = vloadn(i, _ie2);
    REALn ivx = vloadn(i, _ivx);
    REALn ivy = vloadn(i, _ivy);
    REALn ivz = vloadn(i, _ivz);

    REALn ipnax = (REALn)(0);
    REALn ipnay = (REALn)(0);
    REALn ipnaz = (REALn)(0);

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
            pnacc_kernel_core(im, irx, iry, irz, ie2, ivx, ivy, ivz,
                              __jm[k], __jrx[k], __jry[k], __jrz[k],
                              __je2[k], __jvx[k], __jvy[k], __jvz[k],
                              clight,
                              &ipnax, &ipnay, &ipnaz);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (; j < nj; ++j) {
        pnacc_kernel_core(im, irx, iry, irz, ie2, ivx, ivy, ivz,
                          _jm[j], _jrx[j], _jry[j], _jrz[j],
                          _je2[j], _jvx[j], _jvy[j], _jvz[j],
                          clight,
                          &ipnax, &ipnay, &ipnaz);
    }

    vstoren(ipnax, i, _ipnax);
    vstoren(ipnay, i, _ipnay);
    vstoren(ipnaz, i, _ipnaz);
}

