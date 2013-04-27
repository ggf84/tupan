#include "sakura_kernel_common.h"


static inline void
accum_sakura_kernel(
    uint j_begin,
    uint j_end,
    const REAL dt,
    const REAL im,
    const REAL irx,
    const REAL iry,
    const REAL irz,
    const REAL ie2,
    const REAL ivx,
    const REAL ivy,
    const REAL ivz,
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
    REAL *idvx,
    REAL *idvy,
    REAL *idvz)
{
    uint j;
    for (j = j_begin; j < j_end; ++j) {
        REAL jm = __jm[j];
        REAL jrx = __jrx[j];
        REAL jry = __jry[j];
        REAL jrz = __jrz[j];
        REAL je2 = __je2[j];
        REAL jvx = __jvx[j];
        REAL jvy = __jvy[j];
        REAL jvz = __jvz[j];
        sakura_kernel_core(dt,
                           im, irx, iry, irz, ie2, ivx, ivy, ivz,
                           jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
                           &(*idrx), &(*idry), &(*idrz),
                           &(*idvx), &(*idvy), &(*idvz));
    }
}


static inline void
sakura_kernel_main_loop(
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
    REAL *idvx,
    REAL *idvy,
    REAL *idvz)
{
    uint lsize = get_local_size(0);

    uint tile;
    uint numTiles = (nj - 1)/lsize + 1;
    for (tile = 0; tile < numTiles; ++tile) {
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

        uint j = 0;
        uint j_max = (nb > (JUNROLL - 1)) ? (nb - (JUNROLL - 1)):(0);
        for (; j < j_max; j += JUNROLL) {
            accum_sakura_kernel(
                j, j + JUNROLL,
                dt,
                im, irx, iry, irz, ie2, ivx, ivy, ivz,
                __jm, __jrx, __jry, __jrz, __je2, __jvx, __jvy, __jvz,
                &(*idrx), &(*idry), &(*idrz),
                &(*idvx), &(*idvy), &(*idvz));
        }
        accum_sakura_kernel(
            j, nb,
            dt,
            im, irx, iry, irz, ie2, ivx, ivy, ivz,
            __jm, __jrx, __jry, __jrz, __je2, __jvx, __jvy, __jvz,
            &(*idrx), &(*idry), &(*idrz),
            &(*idvx), &(*idvy), &(*idvz));

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}


__kernel void
sakura_kernel(
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
    __local REAL *__jm,
    __local REAL *__jrx,
    __local REAL *__jry,
    __local REAL *__jrz,
    __local REAL *__je2,
    __local REAL *__jvx,
    __local REAL *__jvy,
    __local REAL *__jvz,
    __global REAL *_idrx,
    __global REAL *_idry,
    __global REAL *_idrz,
    __global REAL *_idvx,
    __global REAL *_idvy,
    __global REAL *_idvz)
{
    uint gid = get_global_id(0);
    uint i = (gid < ni) ? (gid) : (ni-1);

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
    REAL idvx = 0;
    REAL idvy = 0;
    REAL idvz = 0;

    sakura_kernel_main_loop(
        dt,
        im, irx, iry, irz, ie2, ivx, ivy, ivz,
        nj,
        _jm, _jrx, _jry, _jrz, _je2, _jvx, _jvy, _jvz,
        __jm, __jrx, __jry, __jrz, __je2, __jvx, __jvy, __jvz,
        &idrx, &idry, &idrz,
        &idvx, &idvy, &idvz);

    _idrx[i] = idrx;
    _idry[i] = idry;
    _idrz[i] = idrz;
    _idvx[i] = idvx;
    _idvy[i] = idvy;
    _idvz[i] = idvz;
}

