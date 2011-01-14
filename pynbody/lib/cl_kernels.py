#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""


from __future__ import (print_function, with_statement)
import pyopencl as cl

from pynbody.lib.decorators import timeit



###########################################################
## CL_KERNEL_0 = {'flops': int, 'name': str, 'source': str}
##
CL_KERNEL_0 = {'flops': 15, 'name': 'set_pot', 'source': """
#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#ifdef cl_amd_fp64
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#define UNROLL_SIZE %(unroll_size)d

typedef %(fp_type)s REAL;
//typedef %(fp_type)s3 REAL3;
typedef %(fp_type)s4 REAL4;


REAL calc_pot(REAL pot, REAL4 bi, REAL4 bj, REAL mj)
{
    REAL4 dr;
    dr.x = bi.x - bj.x;                                              // 1 FLOPs
    dr.y = bi.y - bj.y;                                              // 1 FLOPs
    dr.z = bi.z - bj.z;                                              // 1 FLOPs
    dr.w = bi.w + bj.w;                                              // 1 FLOPs
    REAL ds2 = 0.5 * dr.w;                                           // 1 FLOPs
    ds2 += dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;                  // 6 FLOPs
    REAL rinv = rsqrt(ds2);                                          // 2 FLOPs
    pot -= mj * ((ds2 != 0) ? rinv:0);                               // 2 FLOPs
    return pot;
}


__kernel void set_pot(__global const REAL4 *pos_i,
                      __global const REAL4 *pos_j,
                      __global const REAL *mass_j,
                      __global REAL *pot_i,
                      const uint numBodies,
                      __local REAL4 *pblock,
                      __local REAL *mblock)
{
    uint gwi_idX = get_global_id(0);    uint gwi_sizeX = get_global_size(0);
    uint gwi_idY = get_global_id(1);    uint gwi_sizeY = get_global_size(1);
    uint lwi_idX = get_local_id(0);    uint lwi_sizeX = get_local_size(0);
    uint lwi_idY = get_local_id(1);    uint lwi_sizeY = get_local_size(1);
    uint wg_idX = get_group_id(0);    uint wg_dimX = get_num_groups(0);
    uint wg_idY = get_group_id(1);    uint wg_dimY = get_num_groups(1);


    gwi_idX = UNROLL_SIZE * get_global_id(0);

    REAL4 myPos[UNROLL_SIZE];
    REAL myPot[UNROLL_SIZE];
    for (ushort unroll = 0; unroll < UNROLL_SIZE; unroll++) {
        myPos[unroll] = pos_i[gwi_idX + unroll];
        myPot[unroll] = 0.0;
    }

    for (uint j = 0; j < numBodies; j++) {
        for (ushort unroll = 0; unroll < UNROLL_SIZE; unroll++) {
            myPot[unroll] = calc_pot(myPot[unroll], myPos[unroll],
                                     pos_j[j], mass_j[j]);
        }
    }

    for (ushort unroll = 0; unroll < UNROLL_SIZE; unroll++) {
        pot_i[gwi_idX + unroll] = myPot[unroll];
    }
}

"""}  ## CL_KERNEL_0





##############################################################
## CL_KERNEL_1_DS = {'flops': int, 'name': str, 'source': str}
##
CL_KERNEL_1_DS = {'flops': 57, 'name': 'set_pot', 'source': """
#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#ifdef cl_amd_fp64
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#define UNROLL_SIZE %(unroll_size)d

typedef %(fp_type)s REAL;
//typedef %(fp_type)s3 REAL3;
typedef %(fp_type)s4 REAL4;

typedef float2 DS;
typedef float4 DS2;
typedef float8 DS4;

inline DS to_DS(double a)
{
    DS b;
    b.x = (float)a;
    b.y = (float)(a - b.x);
    return b;
}

inline double to_double(DS a)
{
    double b;
    b = (double)((double)a.x + (double)a.y);
    return b;
}

inline DS dsadd(DS a, DS b)
{
    // Compute dsa + dsb using Knuth's trick.
    float t1 = a.x + b.x;
    float e = t1 - a.x;
    float t2 = ((b.x - e) + (a.x - (t1 - e))) + a.y + b.y;

    // The result is t1 + t2, after normalization.
    DS c;
    c.x = e = t1 + t2;
    c.y = t2 - (e - t1);
    return c;
}

inline float fdsadd(DS a, DS b)
{
    float t1 = a.x + b.x;
    float e = t1 - a.x;
    float t2 = ((b.x - e) + (a.x - (t1 - e))) + a.y + b.y;
    return t1 + t2;
}

DS calc_pot(DS pot, DS4 bi, DS4 bj, DS mj)
{
    float4 dr;
    dr.x = fdsadd(bi.s01, -bj.s01);                                  // 9 FLOPs
    dr.y = fdsadd(bi.s23, -bj.s23);                                  // 9 FLOPs
    dr.z = fdsadd(bi.s45, -bj.s45);                                  // 9 FLOPs
    dr.w = fdsadd(bi.s67, +bj.s67);                                  // 9 FLOPs
    float ds2 = 0.5f * dr.w;                                         // 1 FLOPs
    ds2 += dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;                  // 6 FLOPs
    float rinv = rsqrt(ds2);                                         // 2 FLOPs
    pot = dsadd(pot, -mj * ((ds2 != 0) ? rinv:0));                  // 12 FLOPs
    return pot;
}


__kernel void set_pot(__global const REAL4 *pos_i,
                      __global const REAL4 *pos_j,
                      __global const REAL *mass_j,
                      __global REAL *pot_i,
                      const uint numBodies,
                      __local DS4 *pblock,
                      __local DS *mblock)
{
    uint gwi_idX = get_global_id(0);    uint gwi_sizeX = get_global_size(0);
    uint gwi_idY = get_global_id(1);    uint gwi_sizeY = get_global_size(1);
    uint lwi_idX = get_local_id(0);    uint lwi_sizeX = get_local_size(0);
    uint lwi_idY = get_local_id(1);    uint lwi_sizeY = get_local_size(1);
    uint wg_idX = get_group_id(0);    uint wg_dimX = get_num_groups(0);
    uint wg_idY = get_group_id(1);    uint wg_dimY = get_num_groups(1);


    gwi_idX = UNROLL_SIZE * get_global_id(0);

    DS4 myPos[UNROLL_SIZE];
    DS myPot[UNROLL_SIZE];
    for (ushort unroll = 0; unroll < UNROLL_SIZE; unroll++) {
        REAL4 bi = pos_i[gwi_idX + unroll];
        myPos[unroll].s01 = to_DS(bi.x);
        myPos[unroll].s23 = to_DS(bi.y);
        myPos[unroll].s45 = to_DS(bi.z);
        myPos[unroll].s67 = to_DS(bi.w);
        myPot[unroll] = (DS){0.0, 0.0};
    }

    uint numTiles = numBodies / lwi_sizeX;
    for (uint tile = 0; tile < numTiles; tile++) {

        // load one tile into local memory
        uint jdx = mad24(tile, lwi_sizeX, lwi_idX);

        DS4 otherPos;
        REAL4 bj = pos_j[jdx];
        otherPos.s01 = to_DS(bj.x);
        otherPos.s23 = to_DS(bj.y);
        otherPos.s45 = to_DS(bj.z);
        otherPos.s67 = to_DS(bj.w);

        pblock[lwi_idX] = otherPos;
        mblock[lwi_idX] = to_DS(mass_j[jdx]);

        // Synchronize to make sure data is available for processing
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint k = 0; k < lwi_sizeX; k++) {
            for (ushort unroll = 0; unroll < UNROLL_SIZE; unroll++) {
                myPot[unroll] = calc_pot(myPot[unroll], myPos[unroll],
                                         pblock[k], mblock[k]);
            }
        }

        // Synchronize so that next tile can be loaded
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (ushort unroll = 0; unroll < UNROLL_SIZE; unroll++) {
        pot_i[gwi_idX + unroll] = to_double(myPot[unroll]);
    }
}

"""}  ## CL_KERNEL_1_DS





##############################################################
## P2P_POT_KERNEL = {'flops': int, 'name': str, 'source': str}
##
p2p_pot_kernel = {'flops': 15, 'name': 'p2p_pot_kernel', 'source': """
#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#ifdef cl_amd_fp64
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#ifdef cl_amd_printf
    #pragma OPENCL EXTENSION cl_amd_printf : enable
#endif

#define UNROLL_SIZE %(unroll_size)d

typedef %(fp_type)s REAL;
//typedef %(fp_type)s3 REAL3;
typedef %(fp_type)s4 REAL4;


REAL p2p_pot_kernel_core(REAL pot, REAL4 bi, REAL4 bj, REAL mj)
{
    REAL4 dr;
    dr.x = bi.x - bj.x;                                              // 1 FLOPs
    dr.y = bi.y - bj.y;                                              // 1 FLOPs
    dr.z = bi.z - bj.z;                                              // 1 FLOPs
    dr.w = bi.w + bj.w;                                              // 1 FLOPs
    REAL ds2 = 0.5 * dr.w;                                           // 1 FLOPs
    ds2 += dr.z * dr.z + dr.y * dr.y + dr.x * dr.x;                  // 6 FLOPs
    REAL rinv = rsqrt(ds2);                                          // 2 FLOPs
    pot -= mj * (ds2 ? rinv:0);                                      // 2 FLOPs
    return pot;
}


__kernel void p2p_pot_kernel(__global const REAL4 *ipos,
                             __global const REAL4 *jpos,
                             __global const REAL *jmass,
                             __global REAL *ipot,
                             const uint ni,
                             const uint nj,
                             __local REAL4 *sharedPos,
                             __local REAL *sharedMass)
{
    uint tid = get_local_id(0);
    uint gid = get_global_id(0) * UNROLL_SIZE;
    uint localDim = get_local_size(0);

    REAL4 myPos[UNROLL_SIZE];
    REAL myPot[UNROLL_SIZE];
    for (ushort unroll = 0; unroll < UNROLL_SIZE; unroll++) {
        myPos[unroll] = (gid + unroll < ni) ? ipos[gid + unroll] : ipos[ni-1];
        myPot[unroll] = 0.0;
    }

    uint tile;
    uint numTiles = ((nj + localDim - 1) / localDim) - 1;
    for (tile = 0; tile < numTiles; tile++) {

        uint jdx = tile * localDim + tid;
        sharedPos[tid] = (jdx < nj) ? jpos[jdx] : jpos[nj-1];
        sharedMass[tid] = (jdx < nj) ? jmass[jdx] : jmass[nj-1];

        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint k = 0; k < localDim; k++) {
            for (ushort unroll = 0; unroll < UNROLL_SIZE; unroll++) {
                myPot[unroll] = p2p_pot_kernel_core(myPot[unroll],
                                                    myPos[unroll],
                                                    sharedPos[k],
                                                    sharedMass[k]);
            }
            k++;
            for (ushort unroll = 0; unroll < UNROLL_SIZE; unroll++) {
                myPot[unroll] = p2p_pot_kernel_core(myPot[unroll],
                                                    myPos[unroll],
                                                    sharedPos[k],
                                                    sharedMass[k]);
            }
            k++;
            for (ushort unroll = 0; unroll < UNROLL_SIZE; unroll++) {
                myPot[unroll] = p2p_pot_kernel_core(myPot[unroll],
                                                    myPos[unroll],
                                                    sharedPos[k],
                                                    sharedMass[k]);
            }
            k++;
            for (ushort unroll = 0; unroll < UNROLL_SIZE; unroll++) {
                myPot[unroll] = p2p_pot_kernel_core(myPot[unroll],
                                                    myPos[unroll],
                                                    sharedPos[k],
                                                    sharedMass[k]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    uint jdx = tile * localDim + tid;
    sharedPos[tid] = (jdx < nj) ? jpos[jdx] : jpos[nj-1];
    sharedMass[tid] = (jdx < nj) ? jmass[jdx] : jmass[nj-1];

    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint k = 0; k < nj - (tile * localDim); k++) {
        for (ushort unroll = 0; unroll < UNROLL_SIZE; unroll++) {
                myPot[unroll] = p2p_pot_kernel_core(myPot[unroll],
                                                    myPos[unroll],
                                                    sharedPos[k],
                                                    sharedMass[k]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (ushort unroll = 0; unroll < UNROLL_SIZE; unroll++) {
        if (gid + unroll < ni) {
            ipot[gid + unroll] = myPot[unroll];
        }
    }
}

"""}  ## P2P_POT_KERNEL



UNROLL_SIZE = 5                 # unroll for i-particles
BLOCK_SIZE = (256, 1, 1)
ENABLE_FAST_MATH = True
ENABLE_DOUBLE_PRECISION = True


@timeit
def setup_kernel(source, name):
    """  """    # TODO

    if ENABLE_DOUBLE_PRECISION:
        fp_type = 'double'
    else:
        fp_type = 'float'

    params = {'unroll_size': UNROLL_SIZE,
              'fp_type': fp_type}
    prog = cl.Program(ctx, source % params)

    if ENABLE_FAST_MATH:
        options = '-cl-fast-relaxed-math'
    else:
        options = ' '

    prog.build(options=options)
    kernel = cl.Kernel(prog, name)
    return kernel



ctx = cl.create_some_context()
_properties = cl.command_queue_properties.PROFILING_ENABLE
queue = cl.CommandQueue(ctx, properties=_properties)

# setup the potential kernel
p2p_pot_kernel['kernel'] = setup_kernel(p2p_pot_kernel['source'],
                                        p2p_pot_kernel['name'])

# setup the acceleration kernel
####p2p_acc_kernel['kernel'] = setup_kernel(p2p_acc_kernel['source'],
####                                        p2p_acc_kernel['name'])



########## end of file ##########
