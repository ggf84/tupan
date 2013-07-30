#include "nreg_kernels_common.h"

inline void nreg_Xkernel_main_loop(
    const REAL dt,
    const REAL im,
    const REAL irx,
    const REAL iry,
    const REAL irz,
    const REAL ie2,
    const REAL ivx,
    const REAL ivy,
    const REAL ivz,
    const uint nj,
    __global const REAL *_jm,
    __global const REAL *_jrx,
    __global const REAL *_jry,
    __global const REAL *_jrz,
    __global const REAL *_je2,
    __global const REAL *_jvx,
    __global const REAL *_jvy,
    __global const REAL *_jvz,
    __local REAL *__jm,
    __local REAL *__jrx,
    __local REAL *__jry,
    __local REAL *__jrz,
    __local REAL *__je2,
    __local REAL *__jvx,
    __local REAL *__jvy,
    __local REAL *__jvz,
    REAL *idrx,
    REAL *idry,
    REAL *idrz,
    REAL *iax,
    REAL *iay,
    REAL *iaz,
    REAL *iu)
{
    uint lsize = get_local_size(0);
    uint ntiles = (nj - 1)/lsize + 1;

    for (uint tile = 0; tile < ntiles; ++tile) {
        uint nb = min(lsize, (nj - (tile * lsize)));

        event_t e[8];
        e[0] = async_work_group_copy(__jm,  _jm  + tile * lsize, nb, 0);
        e[1] = async_work_group_copy(__jrx, _jrx + tile * lsize, nb, 0);
        e[2] = async_work_group_copy(__jry, _jry + tile * lsize, nb, 0);
        e[3] = async_work_group_copy(__jrz, _jrz + tile * lsize, nb, 0);
        e[4] = async_work_group_copy(__je2, _je2 + tile * lsize, nb, 0);
        e[5] = async_work_group_copy(__jvx, _jvx + tile * lsize, nb, 0);
        e[6] = async_work_group_copy(__jvy, _jvy + tile * lsize, nb, 0);
        e[7] = async_work_group_copy(__jvz, _jvz + tile * lsize, nb, 0);
        wait_group_events(8, e);

        for (uint j = 0; j < nb; ++j) {
            REAL jm = __jm[j];
            REAL jrx = __jrx[j];
            REAL jry = __jry[j];
            REAL jrz = __jrz[j];
            REAL je2 = __je2[j];
            REAL jvx = __jvx[j];
            REAL jvy = __jvy[j];
            REAL jvz = __jvz[j];
            nreg_Xkernel_core(dt,
                              im, irx, iry, irz, ie2, ivx, ivy, ivz,
                              jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
                              &(*idrx), &(*idry), &(*idrz),
                              &(*iax), &(*iay), &(*iaz), &(*iu));
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}


__kernel void nreg_Xkernel(
    const uint ni,
    __global const REAL *_im,
    __global const REAL *_irx,
    __global const REAL *_iry,
    __global const REAL *_irz,
    __global const REAL *_ie2,
    __global const REAL *_ivx,
    __global const REAL *_ivy,
    __global const REAL *_ivz,
    const uint nj,
    __global const REAL *_jm,
    __global const REAL *_jrx,
    __global const REAL *_jry,
    __global const REAL *_jrz,
    __global const REAL *_je2,
    __global const REAL *_jvx,
    __global const REAL *_jvy,
    __global const REAL *_jvz,
    const REAL dt,
    __global REAL *_idrx,
    __global REAL *_idry,
    __global REAL *_idrz,
    __global REAL *_iax,
    __global REAL *_iay,
    __global REAL *_iaz,
    __global REAL *_iu,
    __local REAL *__jm,
    __local REAL *__jrx,
    __local REAL *__jry,
    __local REAL *__jrz,
    __local REAL *__je2,
    __local REAL *__jvx,
    __local REAL *__jvy,
    __local REAL *__jvz)
{
    uint gid = get_global_id(0);
    uint i = min(gid, ni-1);

    REAL im = _im[i];
    REAL irx = _irx[i];
    REAL iry = _iry[i];
    REAL irz = _irz[i];
    REAL ie2 = _ie2[i];
    REAL ivx = _ivx[i];
    REAL ivy = _ivy[i];
    REAL ivz = _ivz[i];

    REAL idrx = 0;
    REAL idry = 0;
    REAL idrz = 0;
    REAL iax = 0;
    REAL iay = 0;
    REAL iaz = 0;
    REAL iu = 0;

    nreg_Xkernel_main_loop(
        dt,
        im, irx, iry, irz, ie2, ivx, ivy, ivz,
        nj,
        _jm, _jrx, _jry, _jrz, _je2, _jvx, _jvy, _jvz,
        __jm, __jrx, __jry, __jrz, __je2, __jvx, __jvy, __jvz,
        &idrx, &idry, &idrz,
        &iax, &iay, &iaz, &iu);

    _idrx[i] = idrx;
    _idry[i] = idry;
    _idrz[i] = idrz;
    _iax[i] = iax;
    _iay[i] = iay;
    _iaz[i] = iaz;
    _iu[i] = iu;
}



inline void nreg_Vkernel_main_loop(
    const REAL dt,
    const REAL im,
    const REAL ivx,
    const REAL ivy,
    const REAL ivz,
    const REAL iax,
    const REAL iay,
    const REAL iaz,
    const uint nj,
    __global const REAL *_jm,
    __global const REAL *_jvx,
    __global const REAL *_jvy,
    __global const REAL *_jvz,
    __global const REAL *_jax,
    __global const REAL *_jay,
    __global const REAL *_jaz,
    __local REAL *__jm,
    __local REAL *__jvx,
    __local REAL *__jvy,
    __local REAL *__jvz,
    __local REAL *__jax,
    __local REAL *__jay,
    __local REAL *__jaz,
    REAL *idvx,
    REAL *idvy,
    REAL *idvz,
    REAL *ik)
{
    uint lsize = get_local_size(0);
    uint ntiles = (nj - 1)/lsize + 1;

    for (uint tile = 0; tile < ntiles; ++tile) {
        uint nb = min(lsize, (nj - (tile * lsize)));

        event_t e[7];
        e[0] = async_work_group_copy(__jm,  _jm  + tile * lsize, nb, 0);
        e[1] = async_work_group_copy(__jvx, _jvx + tile * lsize, nb, 0);
        e[2] = async_work_group_copy(__jvy, _jvy + tile * lsize, nb, 0);
        e[3] = async_work_group_copy(__jvz, _jvz + tile * lsize, nb, 0);
        e[4] = async_work_group_copy(__jax, _jax + tile * lsize, nb, 0);
        e[5] = async_work_group_copy(__jay, _jay + tile * lsize, nb, 0);
        e[6] = async_work_group_copy(__jaz, _jaz + tile * lsize, nb, 0);
        wait_group_events(7, e);

        for (uint j = 0; j < nb; ++j) {
            REAL jm = __jm[j];
            REAL jvx = __jvx[j];
            REAL jvy = __jvy[j];
            REAL jvz = __jvz[j];
            REAL jax = __jax[j];
            REAL jay = __jay[j];
            REAL jaz = __jaz[j];
            nreg_Vkernel_core(dt,
                              im, ivx, ivy, ivz, iax, iay, iaz,
                              jm, jvx, jvy, jvz, jax, jay, jaz,
                              &(*idvx), &(*idvy), &(*idvz), &(*ik));
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}


__kernel void nreg_Vkernel(
    const uint ni,
    __global const REAL *_im,
    __global const REAL *_ivx,
    __global const REAL *_ivy,
    __global const REAL *_ivz,
    __global const REAL *_iax,
    __global const REAL *_iay,
    __global const REAL *_iaz,
    const uint nj,
    __global const REAL *_jm,
    __global const REAL *_jvx,
    __global const REAL *_jvy,
    __global const REAL *_jvz,
    __global const REAL *_jax,
    __global const REAL *_jay,
    __global const REAL *_jaz,
    const REAL dt,
    __global REAL *_idvx,
    __global REAL *_idvy,
    __global REAL *_idvz,
    __global REAL *_ik,
    __local REAL *__jm,
    __local REAL *__jvx,
    __local REAL *__jvy,
    __local REAL *__jvz,
    __local REAL *__jax,
    __local REAL *__jay,
    __local REAL *__jaz)
{
    uint gid = get_global_id(0);
    uint i = min(gid, ni-1);

    REAL im = _im[i];
    REAL ivx = _ivx[i];
    REAL ivy = _ivy[i];
    REAL ivz = _ivz[i];
    REAL iax = _iax[i];
    REAL iay = _iay[i];
    REAL iaz = _iaz[i];

    REAL idvx = 0;
    REAL idvy = 0;
    REAL idvz = 0;
    REAL ik = 0;

    nreg_Vkernel_main_loop(
        dt,
        im, ivx, ivy, ivz, iax, iay, iaz,
        nj,
        _jm, _jvx, _jvy, _jvz, _jax, _jay, _jaz,
        __jm, __jvx, __jvy, __jvz, __jax, __jay, __jaz,
        &idvx, &idvy, &idvz, &ik);

    _idvx[i] = idvx;
    _idvy[i] = idvy;
    _idvz[i] = idvz;
    _ik[i] = ik;
}

