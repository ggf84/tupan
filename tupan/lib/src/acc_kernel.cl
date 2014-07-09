#include "acc_kernel_common.h"


kernel
__attribute__((reqd_work_group_size(LSIZE, 1, 1)))
void acc_kernel(
    const UINT ni,
    global const REALn * restrict __im,
    global const REALn * restrict __irx,
    global const REALn * restrict __iry,
    global const REALn * restrict __irz,
    global const REALn * restrict __ie2,
    const UINT nj,
    global const REAL * restrict __jm,
    global const REAL * restrict __jrx,
    global const REAL * restrict __jry,
    global const REAL * restrict __jrz,
    global const REAL * restrict __je2,
    global REALn * restrict __iax,
    global REALn * restrict __iay,
    global REALn * restrict __iaz)
{
    for (UINT i = LSIZE * get_group_id(0);
         VW * i < ni; i += LSIZE * get_num_groups(0)) {
        UINT lid = get_local_id(0);
        UINT gid = i + lid;
        gid = ((VW * gid) < ni) ? (gid):(0);

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
        for (; (j + LSIZE - 1) < nj; j += LSIZE) {
            barrier(CLK_LOCAL_MEM_FENCE);
            local REAL _jm[LSIZE];
            local REAL _jrx[LSIZE];
            local REAL _jry[LSIZE];
            local REAL _jrz[LSIZE];
            local REAL _je2[LSIZE];
            _jm[lid] = __jm[j + lid];
            _jrx[lid] = __jrx[j + lid];
            _jry[lid] = __jry[j + lid];
            _jrz[lid] = __jrz[j + lid];
            _je2[lid] = __je2[j + lid];
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll UNROLL
            for (UINT k = 0; k < LSIZE; ++k) {
                acc_kernel_core(
                    im, irx, iry, irz, ie2,
                    _jm[k], _jrx[k], _jry[k], _jrz[k], _je2[k],
                    &iax, &iay, &iaz);
            }
        }
        #endif

        #pragma unroll UNROLL
        for (; j < nj; ++j) {
            acc_kernel_core(
                im, irx, iry, irz, ie2,
                __jm[j], __jrx[j], __jry[j], __jrz[j], __je2[j],
                &iax, &iay, &iaz);
        }

        __iax[gid] = iax;
        __iay[gid] = iay;
        __iaz[gid] = iaz;
    }
}

