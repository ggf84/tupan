#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#define OPENCL_EXTENSION
#include"common.h"
#include"p2p_acc_kernel_core.h"


inline REAL4
p2p_acc_kernel_main_loop(REAL4 myPos,
                         REAL myEps2,
                         const uint nj,
                         __global const REAL4 *jpos,
                         __global const REAL *jeps2,
                         __local REAL4 *sharedPos,
                         __local REAL *sharedEps2)
{
    uint lsize = get_local_size(0);

    REAL4 myAcc = (REAL4){0.0, 0.0, 0.0, 0.0};

    uint tile;
    uint numTiles = (nj - 1)/lsize + 1;
    for (tile = 0; tile < numTiles; ++tile) {
        uint nb = min(lsize, (nj - (tile * lsize)));

        event_t e[2];
        e[0] = async_work_group_copy(sharedPos, jpos + tile * lsize, nb, 0);
        e[1] = async_work_group_copy(sharedEps2, jeps2 + tile * lsize, nb, 0);
        wait_group_events(2, e);

        uint j, jj = 0;
        uint jjmax = (nb > (JUNROLL - 1)) ? (nb - (JUNROLL - 1)):(0);
        for (; jj < jjmax; jj += JUNROLL) {
            for (j = jj; j < jj + JUNROLL; ++j) {
               myAcc = p2p_acc_kernel_core(myAcc,
                                           myPos,
                                           myEps2,
                                           sharedPos[j],
                                           sharedEps2[j]);
            }
        }
        j = jj;
        for (; j < nb; ++j) {
           myAcc = p2p_acc_kernel_core(myAcc,
                                       myPos,
                                       myEps2,
                                       sharedPos[j],
                                       sharedEps2[j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return myAcc;
}


__kernel void p2p_acc_kernel(const uint ni,
                             const uint nj,
                             __global const REAL4 *ipos,
                             __global const REAL *ieps2,
                             __global const REAL4 *jpos,
                             __global const REAL *jeps2,
                             __global REAL4 *iacc,
                             __local REAL4 *sharedPos,
                             __local REAL *sharedEps2)
{
    uint gid = get_global_id(0);
    uint i = (gid < ni) ? (gid) : (ni-1);
    iacc[i] = p2p_acc_kernel_main_loop(ipos[i], ieps2[i],
                                       nj, jpos, jeps2,
                                       sharedPos, sharedEps2);
}

