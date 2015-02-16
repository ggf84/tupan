#include "snap_crackle_kernel_common.h"


__attribute__((reqd_work_group_size(LSIZE, 1, 1)))
kernel void
snap_crackle_kernel(
    uint_t const ni,
    global real_tn const __im[restrict],
    global real_tn const __irx[restrict],
    global real_tn const __iry[restrict],
    global real_tn const __irz[restrict],
    global real_tn const __ie2[restrict],
    global real_tn const __ivx[restrict],
    global real_tn const __ivy[restrict],
    global real_tn const __ivz[restrict],
    global real_tn const __iax[restrict],
    global real_tn const __iay[restrict],
    global real_tn const __iaz[restrict],
    global real_tn const __ijx[restrict],
    global real_tn const __ijy[restrict],
    global real_tn const __ijz[restrict],
    uint_t const nj,
    global real_t const __jm[restrict],
    global real_t const __jrx[restrict],
    global real_t const __jry[restrict],
    global real_t const __jrz[restrict],
    global real_t const __je2[restrict],
    global real_t const __jvx[restrict],
    global real_t const __jvy[restrict],
    global real_t const __jvz[restrict],
    global real_t const __jax[restrict],
    global real_t const __jay[restrict],
    global real_t const __jaz[restrict],
    global real_t const __jjx[restrict],
    global real_t const __jjy[restrict],
    global real_t const __jjz[restrict],
    global real_tn __isx[restrict],
    global real_tn __isy[restrict],
    global real_tn __isz[restrict],
    global real_tn __icx[restrict],
    global real_tn __icy[restrict],
    global real_tn __icz[restrict])
{
    uint_t lid = get_local_id(0);
    uint_t gid = get_global_id(0);
    gid = (gid < ni) ? (gid):(0);

    real_tn im = __im[gid];
    real_tn irx = __irx[gid];
    real_tn iry = __iry[gid];
    real_tn irz = __irz[gid];
    real_tn ie2 = __ie2[gid];
    real_tn ivx = __ivx[gid];
    real_tn ivy = __ivy[gid];
    real_tn ivz = __ivz[gid];
    real_tn iax = __iax[gid];
    real_tn iay = __iay[gid];
    real_tn iaz = __iaz[gid];
    real_tn ijx = __ijx[gid];
    real_tn ijy = __ijy[gid];
    real_tn ijz = __ijz[gid];

    real_tn isx = (real_tn)(0);
    real_tn isy = (real_tn)(0);
    real_tn isz = (real_tn)(0);
    real_tn icx = (real_tn)(0);
    real_tn icy = (real_tn)(0);
    real_tn icz = (real_tn)(0);

    uint_t j = 0;

    #ifdef FAST_LOCAL_MEM
    local real_t _jm[LSIZE];
    local real_t _jrx[LSIZE];
    local real_t _jry[LSIZE];
    local real_t _jrz[LSIZE];
    local real_t _je2[LSIZE];
    local real_t _jvx[LSIZE];
    local real_t _jvy[LSIZE];
    local real_t _jvz[LSIZE];
    local real_t _jax[LSIZE];
    local real_t _jay[LSIZE];
    local real_t _jaz[LSIZE];
    local real_t _jjx[LSIZE];
    local real_t _jjy[LSIZE];
    local real_t _jjz[LSIZE];
    for (; (j + LSIZE - 1) < nj; j += LSIZE) {
        real_t jm = __jm[j + lid];
        real_t jrx = __jrx[j + lid];
        real_t jry = __jry[j + lid];
        real_t jrz = __jrz[j + lid];
        real_t je2 = __je2[j + lid];
        real_t jvx = __jvx[j + lid];
        real_t jvy = __jvy[j + lid];
        real_t jvz = __jvz[j + lid];
        real_t jax = __jax[j + lid];
        real_t jay = __jay[j + lid];
        real_t jaz = __jaz[j + lid];
        real_t jjx = __jjx[j + lid];
        real_t jjy = __jjy[j + lid];
        real_t jjz = __jjz[j + lid];
        barrier(CLK_LOCAL_MEM_FENCE);
        _jm[lid] = jm;
        _jrx[lid] = jrx;
        _jry[lid] = jry;
        _jrz[lid] = jrz;
        _je2[lid] = je2;
        _jvx[lid] = jvx;
        _jvy[lid] = jvy;
        _jvz[lid] = jvz;
        _jax[lid] = jax;
        _jay[lid] = jay;
        _jaz[lid] = jaz;
        _jjx[lid] = jjx;
        _jjy[lid] = jjy;
        _jjz[lid] = jjz;
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll UNROLL
        for (uint_t k = 0; k < LSIZE; ++k) {
            jm = _jm[k];
            jrx = _jrx[k];
            jry = _jry[k];
            jrz = _jrz[k];
            je2 = _je2[k];
            jvx = _jvx[k];
            jvy = _jvy[k];
            jvz = _jvz[k];
            jax = _jax[k];
            jay = _jay[k];
            jaz = _jaz[k];
            jjx = _jjx[k];
            jjy = _jjy[k];
            jjz = _jjz[k];
            snap_crackle_kernel_core(
                im, irx, iry, irz, ie2, ivx, ivy, ivz,
                iax, iay, iaz, ijx, ijy, ijz,
                jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
                jax, jay, jaz, jjx, jjy, jjz,
                &isx, &isy, &isz, &icx, &icy, &icz);
        }
    }
    #endif

    for (; j < nj; ++j) {
        real_t jm = __jm[j];
        real_t jrx = __jrx[j];
        real_t jry = __jry[j];
        real_t jrz = __jrz[j];
        real_t je2 = __je2[j];
        real_t jvx = __jvx[j];
        real_t jvy = __jvy[j];
        real_t jvz = __jvz[j];
        real_t jax = __jax[j];
        real_t jay = __jay[j];
        real_t jaz = __jaz[j];
        real_t jjx = __jjx[j];
        real_t jjy = __jjy[j];
        real_t jjz = __jjz[j];
        snap_crackle_kernel_core(
            im, irx, iry, irz, ie2, ivx, ivy, ivz,
            iax, iay, iaz, ijx, ijy, ijz,
            jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
            jax, jay, jaz, jjx, jjy, jjz,
            &isx, &isy, &isz, &icx, &icy, &icz);
    }

    __isx[gid] = isx;
    __isy[gid] = isy;
    __isz[gid] = isz;
    __icx[gid] = icx;
    __icy[gid] = icy;
    __icz[gid] = icz;
}

