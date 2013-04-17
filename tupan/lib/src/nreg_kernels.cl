#include "nreg_kernels_common.h"


inline REAL
nreg_X_accum(REAL iOmega,
                const REAL8 iData,
                const REAL eta,
                uint j_begin,
                uint j_end,
                __local REAL *sharedJObj_x,
                __local REAL *sharedJObj_y,
                __local REAL *sharedJObj_z,
                __local REAL *sharedJObj_mass,
                __local REAL *sharedJObj_vx,
                __local REAL *sharedJObj_vy,
                __local REAL *sharedJObj_vz,
                __local REAL *sharedJObj_eps2
               )
{
    // ...
    return iOmega;
}


inline REAL
nreg_Xkernel_main_loop(const REAL8 iData,
                           const uint nj,
                           __global const REAL *jobj_x,
                           __global const REAL *jobj_y,
                           __global const REAL *jobj_z,
                           __global const REAL *jobj_mass,
                           __global const REAL *jobj_vx,
                           __global const REAL *jobj_vy,
                           __global const REAL *jobj_vz,
                           __global const REAL *jobj_eps2,
                           const REAL eta,
                           __local REAL *sharedJObj_x,
                           __local REAL *sharedJObj_y,
                           __local REAL *sharedJObj_z,
                           __local REAL *sharedJObj_mass,
                           __local REAL *sharedJObj_vx,
                           __local REAL *sharedJObj_vy,
                           __local REAL *sharedJObj_vz,
                           __local REAL *sharedJObj_eps2
                          )
{
    uint lsize = get_local_size(0);

    REAL iOmega = (REAL)0;

    // ...
    return iOmega;
}


__kernel void nreg_Xkernel(const uint ni,
                               __global const REAL *iobj_x,
                               __global const REAL *iobj_y,
                               __global const REAL *iobj_z,
                               __global const REAL *iobj_mass,
                               __global const REAL *iobj_vx,
                               __global const REAL *iobj_vy,
                               __global const REAL *iobj_vz,
                               __global const REAL *iobj_eps2,
                               const uint nj,
                               __global const REAL *jobj_x,
                               __global const REAL *jobj_y,
                               __global const REAL *jobj_z,
                               __global const REAL *jobj_mass,
                               __global const REAL *jobj_vx,
                               __global const REAL *jobj_vy,
                               __global const REAL *jobj_vz,
                               __global const REAL *jobj_eps2,
                               const REAL eta,
                               __global REAL *itstep,
                               __local REAL *sharedJObj_x,
                               __local REAL *sharedJObj_y,
                               __local REAL *sharedJObj_z,
                               __local REAL *sharedJObj_mass,
                               __local REAL *sharedJObj_vx,
                               __local REAL *sharedJObj_vy,
                               __local REAL *sharedJObj_vz,
                               __local REAL *sharedJObj_eps2
                              )
{
    uint gid = get_global_id(0);
    uint i = (gid < ni) ? (gid) : (ni-1);

    const REAL8 iData = (REAL8){iobj_x[i],
                                iobj_y[i],
                                iobj_z[i],
                                iobj_mass[i],
                                iobj_vx[i],
                                iobj_vy[i],
                                iobj_vz[i],
                                iobj_eps2[i]};

}


__kernel void nreg_Vkernel(const uint ni,
                               __global const REAL *iobj_x,
                               __global const REAL *iobj_y,
                               __global const REAL *iobj_z,
                               __global const REAL *iobj_mass,
                               __global const REAL *iobj_vx,
                               __global const REAL *iobj_vy,
                               __global const REAL *iobj_vz,
                               __global const REAL *iobj_eps2,
                               const uint nj,
                               __global const REAL *jobj_x,
                               __global const REAL *jobj_y,
                               __global const REAL *jobj_z,
                               __global const REAL *jobj_mass,
                               __global const REAL *jobj_vx,
                               __global const REAL *jobj_vy,
                               __global const REAL *jobj_vz,
                               __global const REAL *jobj_eps2,
                               const REAL eta,
                               __global REAL *itstep,
                               __local REAL *sharedJObj_x,
                               __local REAL *sharedJObj_y,
                               __local REAL *sharedJObj_z,
                               __local REAL *sharedJObj_mass,
                               __local REAL *sharedJObj_vx,
                               __local REAL *sharedJObj_vy,
                               __local REAL *sharedJObj_vz,
                               __local REAL *sharedJObj_eps2
                              )
{
    uint gid = get_global_id(0);
    uint i = (gid < ni) ? (gid) : (ni-1);

    const REAL8 iData = (REAL8){iobj_x[i],
                                iobj_y[i],
                                iobj_z[i],
                                iobj_mass[i],
                                iobj_vx[i],
                                iobj_vy[i],
                                iobj_vz[i],
                                iobj_eps2[i]};
}

