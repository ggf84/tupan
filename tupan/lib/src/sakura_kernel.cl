#include "sakura_kernel_common.h"


static inline void
sakura_kernel_accum(
    REAL8 idrdv,
    const REAL8 idata,
    const REAL dt,
    uint j_begin,
    uint j_end,
    REAL3 *idr,
    REAL3 *idv,
    __local REAL *shr_jrx,
    __local REAL *shr_jry,
    __local REAL *shr_jrz,
    __local REAL *shr_jmass,
    __local REAL *shr_jvx,
    __local REAL *shr_jvy,
    __local REAL *shr_jvz,
    __local REAL *shr_jeps2
    )
{
    uint j;
    for (j = j_begin; j < j_end; ++j) {
        REAL8 jdata = (REAL8){shr_jrx[j], shr_jry[j], shr_jrz[j], shr_jmass[j],
                              shr_jvx[j], shr_jvy[j], shr_jvz[j], shr_jeps2[j]};
        sakura_kernel_core(dt,
                           idata.lo, idata.hi,
                           jdata.lo, jdata.hi,
                           idr, idv);
    }
}


static inline void
sakura_kernel_main_loop(
    const REAL8 idata,
    const uint nj,
    __global const REAL *inp_jrx,
    __global const REAL *inp_jry,
    __global const REAL *inp_jrz,
    __global const REAL *inp_jmass,
    __global const REAL *inp_jvx,
    __global const REAL *inp_jvy,
    __global const REAL *inp_jvz,
    __global const REAL *inp_jeps2,
    const REAL dt,
    REAL3 *idr,
    REAL3 *idv,
    __local REAL *shr_jrx,
    __local REAL *shr_jry,
    __local REAL *shr_jrz,
    __local REAL *shr_jmass,
    __local REAL *shr_jvx,
    __local REAL *shr_jvy,
    __local REAL *shr_jvz,
    __local REAL *shr_jeps2
    )
{
    uint lsize = get_local_size(0);

    uint tile;
    uint numTiles = (nj - 1)/lsize + 1;
    for (tile = 0; tile < numTiles; ++tile) {
        uint nb = min(lsize, (nj - (tile * lsize)));

        event_t e[8];
        e[0] = async_work_group_copy(shr_jrx, inp_jrx + tile * lsize, nb, 0);
        e[1] = async_work_group_copy(shr_jry, inp_jry + tile * lsize, nb, 0);
        e[2] = async_work_group_copy(shr_jrz, inp_jrz + tile * lsize, nb, 0);
        e[3] = async_work_group_copy(shr_jmass, inp_jmass + tile * lsize, nb, 0);
        e[4] = async_work_group_copy(shr_jvx, inp_jvx + tile * lsize, nb, 0);
        e[5] = async_work_group_copy(shr_jvy, inp_jvy + tile * lsize, nb, 0);
        e[6] = async_work_group_copy(shr_jvz, inp_jvz + tile * lsize, nb, 0);
        e[7] = async_work_group_copy(shr_jeps2, inp_jeps2 + tile * lsize, nb, 0);
        wait_group_events(8, e);

        uint j = 0;
        uint j_max = (nb > (JUNROLL - 1)) ? (nb - (JUNROLL - 1)):(0);
        for (; j < j_max; j += JUNROLL) {
            sakura_kernel_accum(idata,
                                dt, j, j + JUNROLL,
                                idr, idv,
                                shr_jrx, shr_jry, shr_jrz, shr_jmass,
                                shr_jvx, shr_jvy, shr_jvz, shr_jeps2);
        }
        sakura_kernel_accum(idata,
                            dt, j, nb,
                            idr, idv,
                            shr_jrx, shr_jry, shr_jrz, shr_jmass,
                            shr_jvx, shr_jvy, shr_jvz, shr_jeps2);

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}


__kernel void
sakura_kernel(
    const uint ni,
    __global const REAL *inp_irx,
    __global const REAL *inp_iry,
    __global const REAL *inp_irz,
    __global const REAL *inp_imass,
    __global const REAL *inp_ivx,
    __global const REAL *inp_ivy,
    __global const REAL *inp_ivz,
    __global const REAL *inp_ieps2,
    const uint nj,
    __global const REAL *inp_jrx,
    __global const REAL *inp_jry,
    __global const REAL *inp_jrz,
    __global const REAL *inp_jmass,
    __global const REAL *inp_jvx,
    __global const REAL *inp_jvy,
    __global const REAL *inp_jvz,
    __global const REAL *inp_jeps2,
    const REAL dt,
    __global REAL *out_idrx,
    __global REAL *out_idry,
    __global REAL *out_idrz,
    __global REAL *out_idvx,
    __global REAL *out_idvy,
    __global REAL *out_idvz,
    __local REAL *shr_jrx,
    __local REAL *shr_jry,
    __local REAL *shr_jrz,
    __local REAL *shr_jmass,
    __local REAL *shr_jvx,
    __local REAL *shr_jvy,
    __local REAL *shr_jvz,
    __local REAL *shr_jeps2
    )
{
    uint gid = get_global_id(0);
    uint i = (gid < ni) ? (gid) : (ni-1);

    REAL8 idata = (REAL8){inp_irx[i], inp_iry[i], inp_irz[i], inp_imass[i],
                          inp_ivx[i], inp_ivy[i], inp_ivz[i], inp_ieps2[i]};

    REAL3 idr = (REAL3){0, 0, 0};
    REAL3 idv = (REAL3){0, 0, 0};

    sakura_kernel_main_loop(idata,
                            nj,
                            inp_jrx, inp_jry, inp_jrz, inp_jmass,
                            inp_jvx, inp_jvy, inp_jvz, inp_jeps2,
                            dt,
                            &idr, &idv,
                            shr_jrx, shr_jry, shr_jrz, shr_jmass,
                            shr_jvx, shr_jvy, shr_jvz, shr_jeps2);

    out_idrx[i] = idr.x;
    out_idry[i] = idr.y;
    out_idrz[i] = idr.z;
    out_idvx[i] = idv.x;
    out_idvy[i] = idv.y;
    out_idvz[i] = idv.z;
}

