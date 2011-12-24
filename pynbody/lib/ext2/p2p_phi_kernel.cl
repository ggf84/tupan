#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#define OPENCL_EXTENSION
#include"common.h"
#include"p2p_phi_kernel_core.h"


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

        uint j, jj = 0;
        uint jjmax = (nb > (JUNROLL - 1)) ? (nb - (JUNROLL - 1)):(0);
        for (; jj < jjmax; jj += JUNROLL) {
            for (j = jj; j < jj + JUNROLL; ++j) {
               myPhi = p2p_phi_kernel_core(myPhi,
                                           myPos,
                                           myEps2,
                                           sharedPos[j],
                                           sharedEps2[j]);
            }
        }
            for (j = jj; j < nb; ++j) {
               myPhi = p2p_phi_kernel_core(myPhi,
                                           myPos,
                                           myEps2,
                                           sharedPos[j],
                                           sharedEps2[j]);
            }
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

