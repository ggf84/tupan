#include "tstep_kernel_common.h"


kernel
__attribute__((reqd_work_group_size(LSIZE, 1, 1)))
void tstep_kernel(
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
    const REAL eta,
    global REALn * restrict __idt_a,
    global REALn * restrict __idt_b)
{
    for (UINT i = LSIZE * get_group_id(0);
         VW * i < ni; i += LSIZE * get_num_groups(0)) {
        UINT lid = get_local_id(0);
        UINT gid = ((VW * (i + lid)) < ni) ? (i + lid):(0);

        REALn im = __im[gid];
        REALn irx = __irx[gid];
        REALn iry = __iry[gid];
        REALn irz = __irz[gid];
        REALn ie2 = __ie2[gid];
        REALn ivx = __ivx[gid];
        REALn ivy = __ivy[gid];
        REALn ivz = __ivz[gid];

        REALn iw2_a = (REALn)(0);
        REALn iw2_b = (REALn)(0);

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
                tstep_kernel_core(
                    eta,
                    im, irx, iry, irz, ie2, ivx, ivy, ivz,
                    jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
                    &iw2_a, &iw2_b);
            }
        }
        #endif

        #pragma unroll UNROLL
        for (; j < nj; ++j) {
            REAL jm = __jm[j];
            REAL jrx = __jrx[j];
            REAL jry = __jry[j];
            REAL jrz = __jrz[j];
            REAL je2 = __je2[j];
            REAL jvx = __jvx[j];
            REAL jvy = __jvy[j];
            REAL jvz = __jvz[j];
            tstep_kernel_core(
                eta,
                im, irx, iry, irz, ie2, ivx, ivy, ivz,
                jm, jrx, jry, jrz, je2, jvx, jvy, jvz,
                &iw2_a, &iw2_b);
        }

        __idt_a[gid] = eta / sqrt(fmax((REALn)(1), iw2_a));
        __idt_b[gid] = eta / sqrt(fmax((REALn)(1), iw2_b));
    }
}

