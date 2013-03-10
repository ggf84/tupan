#ifndef P2P_TSTEP_KERNEL_H
#define P2P_TSTEP_KERNEL_H

#include"common.h"
#include"smoothing.h"

//
// p2p_tstep_kernel_core
////////////////////////////////////////////////////////////////////////////////
inline REAL
p2p_tstep_kernel_core(REAL iomega,
                      const REAL4 irm, const REAL4 ive,
                      const REAL4 jrm, const REAL4 jve,
                      const REAL eta)
{
    REAL4 r;
    r.x = irm.x - jrm.x;                                             // 1 FLOPs
    r.y = irm.y - jrm.y;                                             // 1 FLOPs
    r.z = irm.z - jrm.z;                                             // 1 FLOPs
    r.w = irm.w + jrm.w;                                             // 1 FLOPs
    REAL4 v;
    v.x = ive.x - jve.x;                                             // 1 FLOPs
    v.y = ive.y - jve.y;                                             // 1 FLOPs
    v.z = ive.z - jve.z;                                             // 1 FLOPs
    v.w = ive.w + jve.w;                                             // 1 FLOPs
    REAL r2 = r.x * r.x + r.y * r.y + r.z * r.z;                     // 5 FLOPs
    REAL rv = r.x * v.x + r.y * v.y + r.z * v.z;                     // 5 FLOPs
    REAL v2 = v.x * v.x + v.y * v.y + v.z * v.z;                     // 5 FLOPs

    REAL2 ret = smoothed_inv_r2r3(r2, v.w);                          // 5 FLOPs
    REAL inv_r2 = ret.x;
    REAL inv_r3 = ret.y;

    REAL a = 0;
    REAL b = 1;
    REAL c = 2;
    REAL d = 1 / (a + b + c);                                        // 3 FLOPs
    REAL e = (b + c / 2);                                            // 2 FLOPs

    REAL f1 = v2 * inv_r2;                                           // 1 FLOPs
    REAL f2 = r.w * inv_r3;                                          // 1 FLOPs
    REAL omega2 = d * (a + b * f1 + c * f2);                         // 5 FLOPs
    REAL gamma = 1 + d * (e * f2 - a) / omega2;                      // 5 FLOPs
    REAL dln_omega = -gamma * rv * inv_r2;                           // 2 FLOPs
    omega2 = sqrt(omega2);                                           // 1 FLOPs
    omega2 += eta * dln_omega;   // factor 1/2 included in 'eta'     // 2 FLOPs
    omega2 *= omega2;                                                // 1 FLOPs

//    iomega = (omega2 > iomega) ? (omega2):(iomega);
    iomega += (r2 > 0) ? (omega2):(0);                               // 1 FLOPs
    return iomega;
}
// Total flop count: 52


#ifdef __OPENCL_VERSION__
//
// OpenCL implementation
////////////////////////////////////////////////////////////////////////////////
inline REAL
p2p_accum_tstep(REAL iomega,
                const REAL8 idata,
                const REAL eta,
                uint j_begin,
                uint j_end,
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
        iomega = p2p_tstep_kernel_core(iomega,
                                       idata.lo, idata.hi,
                                       jdata.lo, jdata.hi,
                                       eta);
    }
    return iomega;
}


inline REAL
p2p_tstep_kernel_main_loop(const REAL8 idata,
                           const uint nj,
                           __global const REAL *inp_jrx,
                           __global const REAL *inp_jry,
                           __global const REAL *inp_jrz,
                           __global const REAL *inp_jmass,
                           __global const REAL *inp_jvx,
                           __global const REAL *inp_jvy,
                           __global const REAL *inp_jvz,
                           __global const REAL *inp_jeps2,
                           const REAL eta,
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

    REAL iomega = (REAL)0;

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
            iomega = p2p_accum_tstep(iomega, idata,
                                     eta, j, j + JUNROLL,
                                     shr_jrx, shr_jry, shr_jrz, shr_jmass,
                                     shr_jvx, shr_jvy, shr_jvz, shr_jeps2);
        }
        iomega = p2p_accum_tstep(iomega, idata,
                                 eta, j, nb,
                                 shr_jrx, shr_jry, shr_jrz, shr_jmass,
                                 shr_jvx, shr_jvy, shr_jvz, shr_jeps2);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return iomega;
}


__kernel void p2p_tstep_kernel(const uint ni,
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
                               const REAL eta,
                               __global REAL *out_idt,
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

    REAL iomega = p2p_tstep_kernel_main_loop(idata,
                                             nj,
                                             inp_jrx, inp_jry, inp_jrz, inp_jmass,
                                             inp_jvx, inp_jvy, inp_jvz, inp_jeps2,
                                             eta,
                                             shr_jrx, shr_jry, shr_jrz, shr_jmass,
                                             shr_jvx, shr_jvy, shr_jvz, shr_jeps2);

//    out_idt[i] = 2 * eta / iomega;
    out_idt[i] = 2 * eta / sqrt(iomega);
}

#else
//
// C implementation
////////////////////////////////////////////////////////////////////////////////
inline void
p2p_tstep_kernel(const unsigned int ni,
                 const REAL *irx, const REAL *iry, const REAL *irz, const REAL *imass,
                 const REAL *ivx, const REAL *ivy, const REAL *ivz, const REAL *ieps2,
                 const unsigned int nj,
                 const REAL *jrx, const REAL *jry, const REAL *jrz, const REAL *jmass,
                 const REAL *jvx, const REAL *jvy, const REAL *jvz, const REAL *jeps2,
                 const REAL eta,
                 REAL *idt)
{
    unsigned int i, j;
    for (i = 0; i < ni; ++i) {
        REAL4 irm = {irx[i], iry[i], irz[i], imass[i]};
        REAL4 ive = {ivx[i], ivy[i], ivz[i], ieps2[i]};
        REAL iomega = (REAL)0;
        for (j = 0; j < nj; ++j) {
            REAL4 jrm = {jrx[j], jry[j], jrz[j], jmass[j]};
            REAL4 jve = {jvx[j], jvy[j], jvz[j], jeps2[j]};
            iomega = p2p_tstep_kernel_core(iomega, irm, ive, jrm, jve, eta);
        }
//        idt[i] = 2 * eta / iomega;
        idt[i] = 2 * eta / sqrt(iomega);
    }
}

#endif  // __OPENCL_VERSION__
#endif  // P2P_TSTEP_KERNEL_H

