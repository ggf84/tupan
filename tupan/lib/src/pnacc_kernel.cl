#include "pnacc_kernel_common.h"


__attribute__((reqd_work_group_size(LSIZE, 1, 1)))
kernel void
pnacc_kernel(
    uint_t const ni,
    global real_tn const __im[restrict],
    global real_tn const __irx[restrict],
    global real_tn const __iry[restrict],
    global real_tn const __irz[restrict],
    global real_tn const __ie2[restrict],
    global real_tn const __ivx[restrict],
    global real_tn const __ivy[restrict],
    global real_tn const __ivz[restrict],
    uint_t const nj,
    global real_t const __jm[restrict],
    global real_t const __jrx[restrict],
    global real_t const __jry[restrict],
    global real_t const __jrz[restrict],
    global real_t const __je2[restrict],
    global real_t const __jvx[restrict],
    global real_t const __jvy[restrict],
    global real_t const __jvz[restrict],
    constant CLIGHT const * restrict clight,
    global real_tn __ipnax[restrict],
    global real_tn __ipnay[restrict],
    global real_tn __ipnaz[restrict])
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

    real_tn ipnax = (real_tn)(0);
    real_tn ipnay = (real_tn)(0);
    real_tn ipnaz = (real_tn)(0);

    uint_t j = 0;

    #ifdef FAST_LOCAL_MEM
    local real_t _jm[GROUPS][LSIZE];
    local real_t _jrx[GROUPS][LSIZE];
    local real_t _jry[GROUPS][LSIZE];
    local real_t _jrz[GROUPS][LSIZE];
    local real_t _je2[GROUPS][LSIZE];
    local real_t _jvx[GROUPS][LSIZE];
    local real_t _jvy[GROUPS][LSIZE];
    local real_t _jvz[GROUPS][LSIZE];
    #pragma unroll
    for (uint_t g = GROUPS; g > 0; --g) {
        #pragma unroll
        for (; (j + g * LSIZE - 1) < nj; j += g * LSIZE) {
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll
            for (uint_t k = 0; k < g; ++k) {
                _jm[k][lid] = __jm[j + k * LSIZE + lid];
                _jrx[k][lid] = __jrx[j + k * LSIZE + lid];
                _jry[k][lid] = __jry[j + k * LSIZE + lid];
                _jrz[k][lid] = __jrz[j + k * LSIZE + lid];
                _je2[k][lid] = __je2[j + k * LSIZE + lid];
                _jvx[k][lid] = __jvx[j + k * LSIZE + lid];
                _jvy[k][lid] = __jvy[j + k * LSIZE + lid];
                _jvz[k][lid] = __jvz[j + k * LSIZE + lid];
                barrier(CLK_LOCAL_MEM_FENCE);
                #pragma unroll
                for (uint_t l = 0; l < LSIZE; ++l) {
                    real_t jm = _jm[k][l];
                    real_t jrx = _jrx[k][l];
                    real_t jry = _jry[k][l];
                    real_t jrz = _jrz[k][l];
                    real_t je2 = _je2[k][l];
                    real_t jvx = _jvx[k][l];
                    real_t jvy = _jvy[k][l];
                    real_t jvz = _jvz[k][l];
                    pnacc_kernel_core(
                        im, irx, iry, irz, ie2, ivx, ivy, ivz,
                        jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
                        *clight,
                        &ipnax, &ipnay, &ipnaz);
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
        pnacc_kernel_core(
            im, irx, iry, irz, ie2, ivx, ivy, ivz,
            jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
            *clight,
            &ipnax, &ipnay, &ipnaz);
    }

    __ipnax[gid] = ipnax;
    __ipnay[gid] = ipnay;
    __ipnaz[gid] = ipnaz;
}

