#include "pnacc_kernel_common.h"


kernel
__attribute__((reqd_work_group_size(LSIZE, 1, 1)))
void pnacc_kernel(
    const UINT ni,
    global const REALn * restrict __im,
    global const REALn * restrict __irx,
    global const REALn * restrict __iry,
    global const REALn * restrict __irz,
    global const REALn * restrict __ie2,
    global const REALn * restrict __ivx,
    global const REALn * restrict __ivy,
    global const REALn * restrict __ivz,
    const UINT nj,
    global const REAL * restrict __jm,
    global const REAL * restrict __jrx,
    global const REAL * restrict __jry,
    global const REAL * restrict __jrz,
    global const REAL * restrict __je2,
    global const REAL * restrict __jvx,
    global const REAL * restrict __jvy,
    global const REAL * restrict __jvz,
    const UINT order,
    const REAL inv1,
    const REAL inv2,
    const REAL inv3,
    const REAL inv4,
    const REAL inv5,
    const REAL inv6,
    const REAL inv7,
    global REALn * restrict __ipnax,
    global REALn * restrict __ipnay,
    global REALn * restrict __ipnaz)
{
    for (UINT i = LSIZE * get_group_id(0);
         VW * i < ni; i += LSIZE * get_num_groups(0)) {
        UINT lid = get_local_id(0);
        UINT gid = i + lid;
        gid = ((VW * gid) < ni) ? (gid):(0);

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
        for (; (j + LSIZE - 1) < nj; j += LSIZE) {
            barrier(CLK_LOCAL_MEM_FENCE);
            local REAL _jm[LSIZE];
            local REAL _jrx[LSIZE];
            local REAL _jry[LSIZE];
            local REAL _jrz[LSIZE];
            local REAL _je2[LSIZE];
            local REAL _jvx[LSIZE];
            local REAL _jvy[LSIZE];
            local REAL _jvz[LSIZE];
            _jm[lid] = __jm[j + lid];
            _jrx[lid] = __jrx[j + lid];
            _jry[lid] = __jry[j + lid];
            _jrz[lid] = __jrz[j + lid];
            _je2[lid] = __je2[j + lid];
            _jvx[lid] = __jvx[j + lid];
            _jvy[lid] = __jvy[j + lid];
            _jvz[lid] = __jvz[j + lid];
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll UNROLL
            for (UINT k = 0; k < LSIZE; ++k) {
                pnacc_kernel_core(
                    im, irx, iry, irz,
                    ie2, ivx, ivy, ivz,
                    _jm[k], _jrx[k], _jry[k], _jrz[k],
                    _je2[k], _jvx[k], _jvy[k], _jvz[k],
                    clight,
                    &ipnax, &ipnay, &ipnaz);
            }
        }
        #endif

        #pragma unroll UNROLL
        for (; j < nj; ++j) {
            pnacc_kernel_core(
                im, irx, iry, irz,
                ie2, ivx, ivy, ivz,
                __jm[j], __jrx[j], __jry[j], __jrz[j],
                __je2[j], __jvx[j], __jvy[j], __jvz[j],
                clight,
                &ipnax, &ipnay, &ipnaz);
        }

        __ipnax[gid] = ipnax;
        __ipnay[gid] = ipnay;
        __ipnaz[gid] = ipnaz;
    }
}

