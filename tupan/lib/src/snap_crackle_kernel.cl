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
    local real_t _jm[2][LSIZE];
    local real_t _jrx[2][LSIZE];
    local real_t _jry[2][LSIZE];
    local real_t _jrz[2][LSIZE];
    local real_t _je2[2][LSIZE];
    local real_t _jvx[2][LSIZE];
    local real_t _jvy[2][LSIZE];
    local real_t _jvz[2][LSIZE];
    local real_t _jax[2][LSIZE];
    local real_t _jay[2][LSIZE];
    local real_t _jaz[2][LSIZE];
    local real_t _jjx[2][LSIZE];
    local real_t _jjy[2][LSIZE];
    local real_t _jjz[2][LSIZE];
    #pragma unroll
    for (uint_t g = GROUPS; g > 0; g >>= 1) {
        #pragma unroll
        for (; (j + g * LSIZE - 1) < nj; j += g * LSIZE) {
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll
            for (uint_t k = 0; k < g; ++k) {
                _jm[k & 1][lid] = __jm[j + k * LSIZE + lid];
                _jrx[k & 1][lid] = __jrx[j + k * LSIZE + lid];
                _jry[k & 1][lid] = __jry[j + k * LSIZE + lid];
                _jrz[k & 1][lid] = __jrz[j + k * LSIZE + lid];
                _je2[k & 1][lid] = __je2[j + k * LSIZE + lid];
                _jvx[k & 1][lid] = __jvx[j + k * LSIZE + lid];
                _jvy[k & 1][lid] = __jvy[j + k * LSIZE + lid];
                _jvz[k & 1][lid] = __jvz[j + k * LSIZE + lid];
                _jax[k & 1][lid] = __jax[j + k * LSIZE + lid];
                _jay[k & 1][lid] = __jay[j + k * LSIZE + lid];
                _jaz[k & 1][lid] = __jaz[j + k * LSIZE + lid];
                _jjx[k & 1][lid] = __jjx[j + k * LSIZE + lid];
                _jjy[k & 1][lid] = __jjy[j + k * LSIZE + lid];
                _jjz[k & 1][lid] = __jjz[j + k * LSIZE + lid];
                barrier(CLK_LOCAL_MEM_FENCE);
                #pragma unroll
                for (uint_t l = 0; l < LSIZE; ++l) {
                    real_t jm = _jm[k & 1][l];
                    real_t jrx = _jrx[k & 1][l];
                    real_t jry = _jry[k & 1][l];
                    real_t jrz = _jrz[k & 1][l];
                    real_t je2 = _je2[k & 1][l];
                    real_t jvx = _jvx[k & 1][l];
                    real_t jvy = _jvy[k & 1][l];
                    real_t jvz = _jvz[k & 1][l];
                    real_t jax = _jax[k & 1][l];
                    real_t jay = _jay[k & 1][l];
                    real_t jaz = _jaz[k & 1][l];
                    real_t jjx = _jjx[k & 1][l];
                    real_t jjy = _jjy[k & 1][l];
                    real_t jjz = _jjz[k & 1][l];
                    snap_crackle_kernel_core(
                        im, irx, iry, irz, ie2, ivx, ivy, ivz,
                        iax, iay, iaz, ijx, ijy, ijz,
                        jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
                        jax, jay, jaz, jjx, jjy, jjz,
                        &isx, &isy, &isz, &icx, &icy, &icz);
                }
            }
        }
    }
    #endif

    #pragma unroll
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

