#include "acc_kernel_common.h"


__attribute__((reqd_work_group_size(LSIZE, 1, 1)))
kernel void
acc_kernel(
    UINT const ni,
    global REALn const __im[restrict],
    global REALn const __irx[restrict],
    global REALn const __iry[restrict],
    global REALn const __irz[restrict],
    global REALn const __ie2[restrict],
    UINT const nj,
    global REAL const __jm[restrict],
    global REAL const __jrx[restrict],
    global REAL const __jry[restrict],
    global REAL const __jrz[restrict],
    global REAL const __je2[restrict],
    global REALn __iax[restrict],
    global REALn __iay[restrict],
    global REALn __iaz[restrict])
{
    for (UINT i = LSIZE * get_group_id(0) + get_global_offset(0);
              i < ni;
              i += LSIZE * get_num_groups(0)) {
        UINT lid = get_local_id(0);
        UINT gid = ((i + lid) < ni) ? (i + lid):(0);

        REALn im = __im[gid];
        REALn irx = __irx[gid];
        REALn iry = __iry[gid];
        REALn irz = __irz[gid];
        REALn ie2 = __ie2[gid];

        REALn iax = (REALn)(0);
        REALn iay = (REALn)(0);
        REALn iaz = (REALn)(0);

        UINT j = 0;

        #ifdef FAST_LOCAL_MEM
        local REAL _jm[LSIZE];
        local REAL _jrx[LSIZE];
        local REAL _jry[LSIZE];
        local REAL _jrz[LSIZE];
        local REAL _je2[LSIZE];
        for (; (j + LSIZE - 1) < nj; j += LSIZE) {
            REAL jm = __jm[j + lid];
            REAL jrx = __jrx[j + lid];
            REAL jry = __jry[j + lid];
            REAL jrz = __jrz[j + lid];
            REAL je2 = __je2[j + lid];
            barrier(CLK_LOCAL_MEM_FENCE);
            _jm[lid] = jm;
            _jrx[lid] = jrx;
            _jry[lid] = jry;
            _jrz[lid] = jrz;
            _je2[lid] = je2;
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll UNROLL
            for (UINT k = 0; k < LSIZE; ++k) {
                jm = _jm[k];
                jrx = _jrx[k];
                jry = _jry[k];
                jrz = _jrz[k];
                je2 = _je2[k];
                acc_kernel_core(
                    im, irx, iry, irz, ie2,
                    jm, jrx, jry, jrz, je2,
                    &iax, &iay, &iaz);
            }
        }
        #endif

        for (; j < nj; ++j) {
            REAL jm = __jm[j];
            REAL jrx = __jrx[j];
            REAL jry = __jry[j];
            REAL jrz = __jrz[j];
            REAL je2 = __je2[j];
            acc_kernel_core(
                im, irx, iry, irz, ie2,
                jm, jrx, jry, jrz, je2,
                &iax, &iay, &iaz);
        }

        __iax[gid] = iax;
        __iay[gid] = iay;
        __iaz[gid] = iaz;
    }
}

