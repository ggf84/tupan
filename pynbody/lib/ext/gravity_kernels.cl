#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#define OPENCL_EXTENSION
#include"common.h"
#include"p2p_phi_kernel_core.h"
#include"p2p_acc_kernel_core.h"


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


__kernel void p2p_phi_kernel(__global const REAL4 *ipos,
                             __global const REAL *ieps2,
                             __global const REAL4 *jpos,
                             __global const REAL *jeps2,
                             const uint ni,
                             const uint nj,
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
inline REAL4
p2p_accum_acc(REAL4 myAcc,
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
       myAcc = p2p_acc_kernel_core(myAcc, myPos, myVel,
                                   sharedPos[j], sharedVel[j],
                                   eta);
    }
    return myAcc;
}


inline REAL4
p2p_acc_kernel_main_loop(const REAL4 myPos,
                         const REAL4 myVel,
                         __global const REAL4 *jpos,
                         __global const REAL4 *jvel,
                         const uint nj,
                         const REAL eta,
                         __local REAL4 *sharedPos,
                         __local REAL4 *sharedVel)
{
    uint lsize = get_local_size(0);

    REAL4 myAcc = (REAL4){0.0, 0.0, 0.0, 0.0};

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
            myAcc = p2p_accum_acc(myAcc, myPos, myVel,
                                  eta, j, j + JUNROLL,
                                  sharedPos, sharedVel);
        }
        myAcc = p2p_accum_acc(myAcc, myPos, myVel,
                              eta, j, nb,
                              sharedPos, sharedVel);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return myAcc;
}


__kernel void p2p_acc_kernel(__global const REAL4 *ipos,
                             __global const REAL4 *ivel,
                             __global const REAL4 *jpos,
                             __global const REAL4 *jvel,
                             const uint ni,
                             const uint nj,
                             const REAL eta,
                             __global REAL4 *iacc,
                             __local REAL4 *sharedPos,
                             __local REAL4 *sharedVel)
{
    uint gid = get_global_id(0);
    uint i = (gid < ni) ? (gid) : (ni-1);
    iacc[i] = p2p_acc_kernel_main_loop(ipos[i], ivel[i],
                                       jpos, jvel,
                                       nj, eta,
                                       sharedPos, sharedVel);
}

