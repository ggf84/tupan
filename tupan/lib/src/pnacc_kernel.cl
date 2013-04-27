#include "pnacc_kernel_common.h"


inline void
accum_pnacc(
    uint j_begin,
    uint j_end,
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
    const CLIGHT clight,
    REAL *ipnax,
    REAL *ipnay,
    REAL *ipnaz)
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
        pnacc_kernel_core(im, irx, iry, irz, ie2, ivx, ivy, ivz,
                          jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
                          clight,
                          &(*ipnax), &(*ipnay), &(*ipnaz));
    }
}


inline void
pnacc_kernel_main_loop(
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
    const CLIGHT clight,
    REAL *ipnax,
    REAL *ipnay,
    REAL *ipnaz)
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
            accum_pnacc(j, j + JUNROLL,
                        im, irx, iry, irz, ie2, ivx, ivy, ivz,
                        __jm, __jrx, __jry, __jrz, __je2, __jvx, __jvy, __jvz,
                        clight,
                        &(*ipnax), &(*ipnay), &(*ipnaz));
        }
        accum_pnacc(j, nb,
                    im, irx, iry, irz, ie2, ivx, ivy, ivz,
                    __jm, __jrx, __jry, __jrz, __je2, __jvx, __jvy, __jvz,
                    clight,
                    &(*ipnax), &(*ipnay), &(*ipnaz));

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}


__kernel void
pnacc_kernel(
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
    const uint order,
    const REAL inv1,
    const REAL inv2,
    const REAL inv3,
    const REAL inv4,
    const REAL inv5,
    const REAL inv6,
    const REAL inv7,
    __local REAL *__jm,
    __local REAL *__jrx,
    __local REAL *__jry,
    __local REAL *__jrz,
    __local REAL *__je2,
    __local REAL *__jvx,
    __local REAL *__jvy,
    __local REAL *__jvz,
    __global REAL *_ipnax,
    __global REAL *_ipnay,
    __global REAL *_ipnaz)
{
    uint gid = get_global_id(0);
    uint i = (gid < ni) ? (gid) : (ni-1);

    CLIGHT clight = (CLIGHT){.order=order, .inv1=inv1,
                             .inv2=inv2, .inv3=inv3,
                             .inv4=inv4, .inv5=inv5,
                             .inv6=inv6, .inv7=inv7};

    REAL im = _im[i];
    REAL irx = _irx[i];
    REAL iry = _iry[i];
    REAL irz = _irz[i];
    REAL ie2 = _ie2[i];
    REAL ivx = _ivx[i];
    REAL ivy = _ivy[i];
    REAL ivz = _ivz[i];

    REAL ipnax = 0;
    REAL ipnay = 0;
    REAL ipnaz = 0;

    pnacc_kernel_main_loop(
        im, irx, iry, irz, ie2, ivx, ivy, ivz,
        nj,
        _jm, _jrx, _jry, _jrz, _je2, _jvx, _jvy, _jvz,
        __jm, __jrx, __jry, __jrz, __je2, __jvx, __jvy, __jvz,
        clight,
        &ipnax, &ipnay, &ipnaz);

    _ipnax[i] = ipnax;
    _ipnay[i] = ipnay;
    _ipnaz[i] = ipnaz;
}

