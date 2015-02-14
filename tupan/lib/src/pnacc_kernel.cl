#include "pnacc_kernel_common.h"


__attribute__((reqd_work_group_size(LSIZE, 1, 1)))
kernel void
pnacc_kernel(
    UINT const ni,
    global REALn const __im[restrict],
    global REALn const __irx[restrict],
    global REALn const __iry[restrict],
    global REALn const __irz[restrict],
    global REALn const __ie2[restrict],
    global REALn const __ivx[restrict],
    global REALn const __ivy[restrict],
    global REALn const __ivz[restrict],
    UINT const nj,
    global REAL const __jm[restrict],
    global REAL const __jrx[restrict],
    global REAL const __jry[restrict],
    global REAL const __jrz[restrict],
    global REAL const __je2[restrict],
    global REAL const __jvx[restrict],
    global REAL const __jvy[restrict],
    global REAL const __jvz[restrict],
    UINT const order,
    REAL const inv1,
    REAL const inv2,
    REAL const inv3,
    REAL const inv4,
    REAL const inv5,
    REAL const inv6,
    REAL const inv7,
    global REALn __ipnax[restrict],
    global REALn __ipnay[restrict],
    global REALn __ipnaz[restrict])
{
    for (UINT i = LSIZE * get_group_id(0) + get_global_offset(0);
              i < ni;
              i += LSIZE * get_num_groups(0)) {
        UINT lid = get_local_id(0);
        UINT gid = ((i + lid) < ni) ? (i + lid):(0);

        CLIGHT clight = CLIGHT_Init(order, inv1, inv2, inv3,
                                    inv4, inv5, inv6, inv7);

        REALn im = __im[gid];
        REALn irx = __irx[gid];
        REALn iry = __iry[gid];
        REALn irz = __irz[gid];
        REALn ie2 = __ie2[gid];
        REALn ivx = __ivx[gid];
        REALn ivy = __ivy[gid];
        REALn ivz = __ivz[gid];

        REALn ipnax = (REALn)(0);
        REALn ipnay = (REALn)(0);
        REALn ipnaz = (REALn)(0);

        UINT j = 0;

        #ifdef FAST_LOCAL_MEM
        local REAL _jm[LSIZE];
        local REAL _jrx[LSIZE];
        local REAL _jry[LSIZE];
        local REAL _jrz[LSIZE];
        local REAL _je2[LSIZE];
        local REAL _jvx[LSIZE];
        local REAL _jvy[LSIZE];
        local REAL _jvz[LSIZE];
        for (; (j + LSIZE - 1) < nj; j += LSIZE) {
            REAL jm = __jm[j + lid];
            REAL jrx = __jrx[j + lid];
            REAL jry = __jry[j + lid];
            REAL jrz = __jrz[j + lid];
            REAL je2 = __je2[j + lid];
            REAL jvx = __jvx[j + lid];
            REAL jvy = __jvy[j + lid];
            REAL jvz = __jvz[j + lid];
            barrier(CLK_LOCAL_MEM_FENCE);
            _jm[lid] = jm;
            _jrx[lid] = jrx;
            _jry[lid] = jry;
            _jrz[lid] = jrz;
            _je2[lid] = je2;
            _jvx[lid] = jvx;
            _jvy[lid] = jvy;
            _jvz[lid] = jvz;
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll UNROLL
            for (UINT k = 0; k < LSIZE; ++k) {
                jm = _jm[k];
                jrx = _jrx[k];
                jry = _jry[k];
                jrz = _jrz[k];
                je2 = _je2[k];
                jvx = _jvx[k];
                jvy = _jvy[k];
                jvz = _jvz[k];
                pnacc_kernel_core(
                    im, irx, iry, irz, ie2, ivx, ivy, ivz,
                    jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
                    clight,
                    &ipnax, &ipnay, &ipnaz);
            }
        }
        #endif

        for (; j < nj; ++j) {
            REAL jm = __jm[j];
            REAL jrx = __jrx[j];
            REAL jry = __jry[j];
            REAL jrz = __jrz[j];
            REAL je2 = __je2[j];
            REAL jvx = __jvx[j];
            REAL jvy = __jvy[j];
            REAL jvz = __jvz[j];
            pnacc_kernel_core(
                im, irx, iry, irz, ie2, ivx, ivy, ivz,
                jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
                clight,
                &ipnax, &ipnay, &ipnaz);
        }

        __ipnax[gid] = ipnax;
        __ipnay[gid] = ipnay;
        __ipnaz[gid] = ipnaz;
    }
}

