#include "snap_crackle_kernel_common.h"

inline void snap_crackle_kernel_main_loop(
    const REAL im,
    const REAL irx,
    const REAL iry,
    const REAL irz,
    const REAL ie2,
    const REAL ivx,
    const REAL ivy,
    const REAL ivz,
    const REAL iax,
    const REAL iay,
    const REAL iaz,
    const REAL ijx,
    const REAL ijy,
    const REAL ijz,
    const uint nj,
    __global const REAL *_jm,
    __global const REAL *_jrx,
    __global const REAL *_jry,
    __global const REAL *_jrz,
    __global const REAL *_je2,
    __global const REAL *_jvx,
    __global const REAL *_jvy,
    __global const REAL *_jvz,
    __global const REAL *_jax,
    __global const REAL *_jay,
    __global const REAL *_jaz,
    __global const REAL *_jjx,
    __global const REAL *_jjy,
    __global const REAL *_jjz,
    __local REAL *__jm,
    __local REAL *__jrx,
    __local REAL *__jry,
    __local REAL *__jrz,
    __local REAL *__je2,
    __local REAL *__jvx,
    __local REAL *__jvy,
    __local REAL *__jvz,
    __local REAL *__jax,
    __local REAL *__jay,
    __local REAL *__jaz,
    __local REAL *__jjx,
    __local REAL *__jjy,
    __local REAL *__jjz,
    REAL *isx,
    REAL *isy,
    REAL *isz,
    REAL *icx,
    REAL *icy,
    REAL *icz)
{
    uint lsize = get_local_size(0);
    uint ntiles = (nj - 1)/lsize + 1;

    for (uint tile = 0; tile < ntiles; ++tile) {
        uint nb = min(lsize, (nj - (tile * lsize)));

        event_t e[14];
        e[0] = async_work_group_copy(__jm,  _jm  + tile * lsize, nb, 0);
        e[1] = async_work_group_copy(__jrx, _jrx + tile * lsize, nb, 0);
        e[2] = async_work_group_copy(__jry, _jry + tile * lsize, nb, 0);
        e[3] = async_work_group_copy(__jrz, _jrz + tile * lsize, nb, 0);
        e[4] = async_work_group_copy(__je2, _je2 + tile * lsize, nb, 0);
        e[5] = async_work_group_copy(__jvx, _jvx + tile * lsize, nb, 0);
        e[6] = async_work_group_copy(__jvy, _jvy + tile * lsize, nb, 0);
        e[7] = async_work_group_copy(__jvz, _jvz + tile * lsize, nb, 0);
        e[8] = async_work_group_copy(__jax, _jax + tile * lsize, nb, 0);
        e[9] = async_work_group_copy(__jay, _jay + tile * lsize, nb, 0);
        e[10] = async_work_group_copy(__jaz, _jaz + tile * lsize, nb, 0);
        e[11] = async_work_group_copy(__jjx, _jjx + tile * lsize, nb, 0);
        e[12] = async_work_group_copy(__jjy, _jjy + tile * lsize, nb, 0);
        e[13] = async_work_group_copy(__jjz, _jjz + tile * lsize, nb, 0);
        wait_group_events(14, e);

        for (uint j = 0; j < nb; ++j) {
            REAL jm = __jm[j];
            REAL jrx = __jrx[j];
            REAL jry = __jry[j];
            REAL jrz = __jrz[j];
            REAL je2 = __je2[j];
            REAL jvx = __jvx[j];
            REAL jvy = __jvy[j];
            REAL jvz = __jvz[j];
            REAL jax = __jax[j];
            REAL jay = __jay[j];
            REAL jaz = __jaz[j];
            REAL jjx = __jjx[j];
            REAL jjy = __jjy[j];
            REAL jjz = __jjz[j];
            snap_crackle_kernel_core(im, irx, iry, irz,
                                     ie2, ivx, ivy, ivz,
                                     iax, iay, iaz, ijx, ijy, ijz,
                                     jm, jrx, jry, jrz,
                                     je2, jvx, jvy, jvz,
                                     jax, jay, jaz, jjx, jjy, jjz,
                                     &(*isx), &(*isy), &(*isz),
                                     &(*icx), &(*icy), &(*icz));
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}


__kernel void snap_crackle_kernel(
    const uint ni,
    __global const REAL *_im,
    __global const REAL *_irx,
    __global const REAL *_iry,
    __global const REAL *_irz,
    __global const REAL *_ie2,
    __global const REAL *_ivx,
    __global const REAL *_ivy,
    __global const REAL *_ivz,
    __global const REAL *_iax,
    __global const REAL *_iay,
    __global const REAL *_iaz,
    __global const REAL *_ijx,
    __global const REAL *_ijy,
    __global const REAL *_ijz,
    const uint nj,
    __global const REAL *_jm,
    __global const REAL *_jrx,
    __global const REAL *_jry,
    __global const REAL *_jrz,
    __global const REAL *_je2,
    __global const REAL *_jvx,
    __global const REAL *_jvy,
    __global const REAL *_jvz,
    __global const REAL *_jax,
    __global const REAL *_jay,
    __global const REAL *_jaz,
    __global const REAL *_jjx,
    __global const REAL *_jjy,
    __global const REAL *_jjz,
    __global REAL *_isx,
    __global REAL *_isy,
    __global REAL *_isz,
    __global REAL *_icx,
    __global REAL *_icy,
    __global REAL *_icz,
    __local REAL *__jm,
    __local REAL *__jrx,
    __local REAL *__jry,
    __local REAL *__jrz,
    __local REAL *__je2,
    __local REAL *__jvx,
    __local REAL *__jvy,
    __local REAL *__jvz,
    __local REAL *__jax,
    __local REAL *__jay,
    __local REAL *__jaz,
    __local REAL *__jjx,
    __local REAL *__jjy,
    __local REAL *__jjz)
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
    REAL iax = _iax[i];
    REAL iay = _iay[i];
    REAL iaz = _iaz[i];
    REAL ijx = _ijx[i];
    REAL ijy = _ijy[i];
    REAL ijz = _ijz[i];

    REAL isx = 0;
    REAL isy = 0;
    REAL isz = 0;
    REAL icx = 0;
    REAL icy = 0;
    REAL icz = 0;

    snap_crackle_kernel_main_loop(
        im, irx, iry, irz,
        ie2, ivx, ivy, ivz,
        iax, iay, iaz, ijx, ijy, ijz,
        nj,
        _jm, _jrx, _jry, _jrz,
        _je2, _jvx, _jvy, _jvz,
        _jax, _jay, _jaz, _jjx, _jjy, _jjz,
        __jm, __jrx, __jry, __jrz,
        __je2, __jvx, __jvy, __jvz,
        __jax, __jay, __jaz, __jjx, __jjy, __jjz,
        &isx, &isy, &isz,
        &icx, &icy, &icz);

    _isx[i] = isx;
    _isy[i] = isy;
    _isz[i] = isz;
    _icx[i] = icx;
    _icy[i] = icy;
    _icz[i] = icz;
}

