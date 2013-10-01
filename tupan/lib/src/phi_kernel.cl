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
    __global REAL * restrict _iphi,
    __local REAL *__jm,
    __local REAL *__jrx,
    __local REAL *__jry,
    __local REAL *__jrz,
    __local REAL *__je2)
{
    UINT i = get_global_id(0);

    REALn im = vloadn(i, _im);
    REALn irx = vloadn(i, _irx);
    REALn iry = vloadn(i, _iry);
    REALn irz = vloadn(i, _irz);
    REALn ie2 = vloadn(i, _ie2);

    REALn iphi = (REALn)(0);

    UINT j = 0;
    for (; (j + LSIZE) < nj; j += LSIZE) {
        event_t e[5];
        e[0] = async_work_group_copy(__jm, _jm + j, LSIZE, 0);
        e[1] = async_work_group_copy(__jrx, _jrx + j, LSIZE, 0);
        e[2] = async_work_group_copy(__jry, _jry + j, LSIZE, 0);
        e[3] = async_work_group_copy(__jrz, _jrz + j, LSIZE, 0);
        e[4] = async_work_group_copy(__je2, _je2 + j, LSIZE, 0);
        wait_group_events(5, e);
        for (UINT k = 0; k < LSIZE; ++k) {
            phi_kernel_core(im, irx, iry, irz, ie2,
                            __jm[k], __jrx[k], __jry[k], __jrz[k], __je2[k],
                            &iphi);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (; j < nj; ++j) {
        phi_kernel_core(im, irx, iry, irz, ie2,
                        _jm[j], _jrx[j], _jry[j], _jrz[j], _je2[j],
                        &iphi);
    }

    vstoren(iphi, i, _iphi);
}

