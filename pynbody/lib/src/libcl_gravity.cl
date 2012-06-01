#if __OPENCL_VERSION__ <= CL_VERSION_1_1
    #ifdef cl_khr_fp64
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #endif
#endif

#include"common.h"
#include"gravity_kernels.h"


//
// Phi methods
////////////////////////////////////////////////////////////////////////////////
inline REAL
p2p_accum_phi(REAL myPhi,
              const REAL4 myPos,
              const REAL myEps2,
              uint j_begin,
              uint j_end,
              __local REAL4 *sharedPos,
              __local REAL *sharedEps2)
{
    uint j;
    for (j = j_begin; j < j_end; ++j) {
       myPhi = p2p_phi_kernel_core(myPhi, myPos, myEps2,
                                   sharedPos[j], sharedEps2[j]);

    }
    return myPhi;
}


inline REAL
p2p_phi_kernel_main_loop(const REAL4 myPos,
                         const REAL myEps2,
                         __global const REAL4 *jpos,
                         __global const REAL *jeps2,
                         const uint nj,
                         __local REAL4 *sharedPos,
                         __local REAL *sharedEps2)
{
    uint lsize = get_local_size(0);

    REAL myPhi = (REAL)0.0;

    uint tile;
    uint numTiles = (nj - 1)/lsize + 1;
    for (tile = 0; tile < numTiles; ++tile) {
        uint nb = min(lsize, (nj - (tile * lsize)));

        event_t e[2];
        e[0] = async_work_group_copy(sharedPos, jpos + tile * lsize, nb, 0);
        e[1] = async_work_group_copy(sharedEps2, jeps2 + tile * lsize, nb, 0);
        wait_group_events(2, e);

        uint j = 0;
        uint j_max = (nb > (JUNROLL - 1)) ? (nb - (JUNROLL - 1)):(0);
        for (; j < j_max; j += JUNROLL) {
            myPhi = p2p_accum_phi(myPhi, myPos, myEps2,
                                  j, j + JUNROLL,
                                  sharedPos, sharedEps2);
        }
        myPhi = p2p_accum_phi(myPhi, myPos, myEps2,
                              j, nb,
                              sharedPos, sharedEps2);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return myPhi;
}


__kernel void p2p_phi_kernel(const uint ni,
                             __global const REAL4 *ipos,
                             __global const REAL *ieps2,
                             const uint nj,
                             __global const REAL4 *jpos,
                             __global const REAL *jeps2,
                             __global REAL *iphi,
                             __local REAL4 *sharedPos,
                             __local REAL *sharedEps2)
{
    uint gid = get_global_id(0);
    uint i = (gid < ni) ? (gid) : (ni-1);
    iphi[i] = p2p_phi_kernel_main_loop(ipos[i], ieps2[i],
                                       jpos, jeps2,
                                       nj,
                                       sharedPos, sharedEps2);
}


//
// Acc methods
////////////////////////////////////////////////////////////////////////////////
inline REAL3
p2p_accum_acc(REAL3 myAcc,
              const REAL4 myPos,
              const REAL myEps2,
              uint j_begin,
              uint j_end,
              __local REAL4 *sharedPos,
              __local REAL *sharedEps2)
{
    uint j;
    for (j = j_begin; j < j_end; ++j) {
       myAcc = p2p_acc_kernel_core(myAcc, myPos, myEps2,
                                   sharedPos[j], sharedEps2[j]);
    }
    return myAcc;
}


inline REAL4
p2p_acc_kernel_main_loop(const REAL4 myPos,
                         const REAL myEps2,
                         __global const REAL4 *jpos,
                         __global const REAL *jeps2,
                         const uint nj,
                         __local REAL4 *sharedPos,
                         __local REAL *sharedEps2)
{
    uint lsize = get_local_size(0);

    REAL3 myAcc = (REAL3){0.0, 0.0, 0.0};

    uint tile;
    uint numTiles = (nj - 1)/lsize + 1;
    for (tile = 0; tile < numTiles; ++tile) {
        uint nb = min(lsize, (nj - (tile * lsize)));

        event_t e[2];
        e[0] = async_work_group_copy(sharedPos, jpos + tile * lsize, nb, 0);
        e[1] = async_work_group_copy(sharedEps2, jeps2 + tile * lsize, nb, 0);
        wait_group_events(2, e);

        uint j = 0;
        uint j_max = (nb > (JUNROLL - 1)) ? (nb - (JUNROLL - 1)):(0);
        for (; j < j_max; j += JUNROLL) {
            myAcc = p2p_accum_acc(myAcc, myPos, myEps2,
                                  j, j + JUNROLL,
                                  sharedPos, sharedEps2);
        }
        myAcc = p2p_accum_acc(myAcc, myPos, myEps2,
                              j, nb,
                              sharedPos, sharedEps2);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return (REAL4){myAcc.x, myAcc.y, myAcc.z, 0.0};
}


__kernel void p2p_acc_kernel(const uint ni,
                             __global const REAL4 *ipos,
                             __global const REAL *ieps2,
                             const uint nj,
                             __global const REAL4 *jpos,
                             __global const REAL *jeps2,
                             __global REAL4 *iacc,  // XXX: Bug!!! if we use __global REAL3
                             __local REAL4 *sharedPos,
                             __local REAL *sharedEps2)
{
    uint gid = get_global_id(0);
    uint i = (gid < ni) ? (gid) : (ni-1);
    iacc[i] = p2p_acc_kernel_main_loop(ipos[i], ieps2[i],
                                       jpos, jeps2,
                                       nj,
                                       sharedPos, sharedEps2);
}


//
// AccTstep methods
////////////////////////////////////////////////////////////////////////////////
inline REAL4
p2p_accum_acctstep(REAL4 myAccTstep,
                   const REAL4 myPos,
                   const REAL4 myVel,
                   const REAL eta,
                   uint j_begin,
                   uint j_end,
                   __local REAL4 *sharedPos,
                   __local REAL4 *sharedVel)
{
    uint j;
    for (j = j_begin; j < j_end; ++j) {
       myAccTstep = p2p_acctstep_kernel_core(myAccTstep, myPos, myVel,
                                             sharedPos[j], sharedVel[j],
                                             eta);
    }
    return myAccTstep;
}


inline REAL4
p2p_acctstep_kernel_main_loop(const REAL4 myPos,
                              const REAL4 myVel,
                              __global const REAL4 *jpos,
                              __global const REAL4 *jvel,
                              const uint nj,
                              const REAL eta,
                              __local REAL4 *sharedPos,
                              __local REAL4 *sharedVel)
{
    uint lsize = get_local_size(0);

    REAL4 myAccTstep = (REAL4){0.0, 0.0, 0.0, 0.0};

    uint tile;
    uint numTiles = (nj - 1)/lsize + 1;
    for (tile = 0; tile < numTiles; ++tile) {
        uint nb = min(lsize, (nj - (tile * lsize)));

        event_t e[2];
        e[0] = async_work_group_copy(sharedPos, jpos + tile * lsize, nb, 0);
        e[1] = async_work_group_copy(sharedVel, jvel + tile * lsize, nb, 0);
        wait_group_events(2, e);

        uint j = 0;
        uint j_max = (nb > (JUNROLL - 1)) ? (nb - (JUNROLL - 1)):(0);
        for (; j < j_max; j += JUNROLL) {
            myAccTstep = p2p_accum_acctstep(myAccTstep, myPos, myVel,
                                            eta, j, j + JUNROLL,
                                            sharedPos, sharedVel);
        }
        myAccTstep = p2p_accum_acctstep(myAccTstep, myPos, myVel,
                                        eta, j, nb,
                                        sharedPos, sharedVel);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return myAccTstep;
}


__kernel void p2p_acctstep_kernel(const uint ni,
                                  __global const REAL4 *ipos,
                                  __global const REAL4 *ivel,
                                  const uint nj,
                                  __global const REAL4 *jpos,
                                  __global const REAL4 *jvel,
                                  const REAL eta,
                                  __global REAL4 *iacctstep,
                                  __local REAL4 *sharedPos,
                                  __local REAL4 *sharedVel)
{
    uint gid = get_global_id(0);
    uint i = (gid < ni) ? (gid) : (ni-1);
    iacctstep[i] = p2p_acctstep_kernel_main_loop(ipos[i], ivel[i],
                                                 jpos, jvel,
                                                 nj, eta,
                                                 sharedPos, sharedVel);
}


//
// Tstep methods
////////////////////////////////////////////////////////////////////////////////
inline REAL
p2p_accum_tstep(REAL myInvTstep,
                const REAL8 myData,
                const REAL eta,
                uint j_begin,
                uint j_end,
                __local REAL8 *sharedJData
                )
{
    uint j;
    for (j = j_begin; j < j_end; ++j) {
       myInvTstep = p2p_tstep_kernel_core(myInvTstep, myData.lo, myData.hi,
                                          sharedJData[j].lo, sharedJData[j].hi,
                                          eta);
    }
    return myInvTstep;
}


inline REAL
p2p_tstep_kernel_main_loop(const REAL8 myData,
                           const uint nj,
                           __global const REAL8 *jdata,
                           const REAL eta,
                           __local REAL8 *sharedJData
                          )
{
    uint lsize = get_local_size(0);

    REAL myInvTstep = (REAL)0.0;

    uint tile;
    uint numTiles = (nj - 1)/lsize + 1;
    for (tile = 0; tile < numTiles; ++tile) {
        uint nb = min(lsize, (nj - (tile * lsize)));

        event_t e[1];
        e[0] = async_work_group_copy(sharedJData, jdata + tile * lsize, nb, 0);
        wait_group_events(1, e);

        uint j = 0;
        uint j_max = (nb > (JUNROLL - 1)) ? (nb - (JUNROLL - 1)):(0);
        for (; j < j_max; j += JUNROLL) {
            myInvTstep = p2p_accum_tstep(myInvTstep, myData,
                                         eta, j, j + JUNROLL,
                                         sharedJData);
        }
        myInvTstep = p2p_accum_tstep(myInvTstep, myData,
                                     eta, j, nb,
                                     sharedJData);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return myInvTstep;
}


__kernel void p2p_tstep_kernel(const uint ni,
                               __global const REAL8 *idata,
                               const uint nj,
                               __global const REAL8 *jdata,
                               const REAL eta,
                               __global REAL *iinv_tstep,
                               __local REAL8 *sharedJData
                              )
{
    uint gid = get_global_id(0);
    uint i = (gid < ni) ? (gid) : (ni-1);
    iinv_tstep[i] = p2p_tstep_kernel_main_loop(idata[i],
                                               nj, jdata,
                                               eta,
                                               sharedJData);
}


//
// PN-Acc methods (experimental)
////////////////////////////////////////////////////////////////////////////////
inline REAL3
p2p_accum_pnacc(REAL3 myPNAcc,
                const REAL4 myPos,
                const REAL4 myVel,
                const CLIGHT clight,
                uint j_begin,
                uint j_end,
                __local REAL4 *sharedPos,
                __local REAL4 *sharedVel)
{
    uint j;
    for (j = j_begin; j < j_end; ++j) {
       myPNAcc = p2p_pnacc_kernel_core(myPNAcc, myPos, myVel,
                                       sharedPos[j], sharedVel[j],
                                       clight);
    }
    return myPNAcc;
}


inline REAL4
p2p_pnacc_kernel_main_loop(const REAL4 myPos,
                           const REAL4 myVel,
                           __global const REAL4 *jpos,
                           __global const REAL4 *jvel,
                           const uint nj,
                           const CLIGHT clight,
                           __local REAL4 *sharedPos,
                           __local REAL4 *sharedVel)
{
    uint lsize = get_local_size(0);

    REAL3 myPNAcc = (REAL3){0.0, 0.0, 0.0};

    uint tile;
    uint numTiles = (nj - 1)/lsize + 1;
    for (tile = 0; tile < numTiles; ++tile) {
        uint nb = min(lsize, (nj - (tile * lsize)));

        event_t e[2];
        e[0] = async_work_group_copy(sharedPos, jpos + tile * lsize, nb, 0);
        e[1] = async_work_group_copy(sharedVel, jvel + tile * lsize, nb, 0);
        wait_group_events(2, e);

        uint j = 0;
        uint j_max = (nb > (JUNROLL - 1)) ? (nb - (JUNROLL - 1)):(0);
        for (; j < j_max; j += JUNROLL) {
            myPNAcc = p2p_accum_pnacc(myPNAcc, myPos, myVel,
                                      clight, j, j + JUNROLL,
                                      sharedPos, sharedVel);
        }
        myPNAcc = p2p_accum_pnacc(myPNAcc, myPos, myVel,
                                  clight, j, nb,
                                  sharedPos, sharedVel);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return (REAL4){myPNAcc.x, myPNAcc.y, myPNAcc.z, 0.0};
}


__kernel void p2p_pnacc_kernel(const uint ni,
                               __global const REAL4 *ipos,
                               __global const REAL4 *ivel,
                               const uint nj,
                               __global const REAL4 *jpos,
                               __global const REAL4 *jvel,
                               const uint order,
                               const REAL cinv1,
                               const REAL cinv2,
                               const REAL cinv3,
                               const REAL cinv4,
                               const REAL cinv5,
                               const REAL cinv6,
                               const REAL cinv7,
                               __global REAL4 *ipnacc,  // XXX: Bug!!! if we use __global REAL3
                               __local REAL4 *sharedPos,
                               __local REAL4 *sharedVel)
{
    uint gid = get_global_id(0);
    uint i = (gid < ni) ? (gid) : (ni-1);
    const CLIGHT clight = (CLIGHT){cinv1, cinv2, cinv3,
                                   cinv4, cinv5, cinv6,
                                   cinv7, order};
    ipnacc[i] = p2p_pnacc_kernel_main_loop(ipos[i], ivel[i],
                                           jpos, jvel,
                                           nj, clight,
                                           sharedPos, sharedVel);
}

