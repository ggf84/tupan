#include "phi_kernel_common.h"

inline void phi_kernel_main_loop(
    const REAL im,
    const REAL irx,
    const REAL iry,
    const REAL irz,
    const REAL ie2,
    const UINT nj,
    __global const REAL *_jm,
    __global const REAL *_jrx,
    __global const REAL *_jry,
    __global const REAL *_jrz,
    __global const REAL *_je2,
    __local REAL *__jm,
    __local REAL *__jrx,
    __local REAL *__jry,
    __local REAL *__jrz,
    __local REAL *__je2,
    REAL *iphi)
{
    UINT lsize = get_local_size(0);
    UINT ntiles = (nj - 1)/lsize + 1;

    for (UINT tile = 0; tile < ntiles; ++tile) {
        UINT nb = min(lsize, (nj - (tile * lsize)));

        event_t e[5];
        e[0] = async_work_group_copy(__jm,  _jm  + tile * lsize, nb, 0);
        e[1] = async_work_group_copy(__jrx, _jrx + tile * lsize, nb, 0);
        e[2] = async_work_group_copy(__jry, _jry + tile * lsize, nb, 0);
        e[3] = async_work_group_copy(__jrz, _jrz + tile * lsize, nb, 0);
        e[4] = async_work_group_copy(__je2, _je2 + tile * lsize, nb, 0);
        wait_group_events(5, e);

        for (UINT j = 0; j < nb; ++j) {
            REAL jm = __jm[j];
            REAL jrx = __jrx[j];
            REAL jry = __jry[j];
            REAL jrz = __jrz[j];
            REAL je2 = __je2[j];
            phi_kernel_core(im, irx, iry, irz, ie2,
                            jm, jrx, jry, jrz, je2,
                            &(*iphi));
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}


__kernel void phi_kernel(
    const UINT ni,
    __global const REAL *_im,
    __global const REAL *_irx,
    __global const REAL *_iry,
    __global const REAL *_irz,
    __global const REAL *_ie2,
    const UINT nj,
    __global const REAL *_jm,
    __global const REAL *_jrx,
    __global const REAL *_jry,
    __global const REAL *_jrz,
    __global const REAL *_je2,
    __global REAL *_iphi,
    __local REAL *__jm,
    __local REAL *__jrx,
    __local REAL *__jry,
    __local REAL *__jrz,
    __local REAL *__je2)
{
    UINT gid = get_global_id(0);
    UINT i = min(gid, ni-1);

    REAL im = _im[i];
    REAL irx = _irx[i];
    REAL iry = _iry[i];
    REAL irz = _irz[i];
    REAL ie2 = _ie2[i];

    REAL iphi = 0;

    phi_kernel_main_loop(
        im, irx, iry, irz, ie2,
        nj,
        _jm, _jrx, _jry, _jrz, _je2,
        __jm, __jrx, __jry, __jrz, __je2,
        &iphi);

    _iphi[i] = iphi;
}

