#ifndef P2P_TSTEP_KERNEL_H
#define P2P_TSTEP_KERNEL_H

#include"common.h"
#include"smoothing.h"

//
// p2p_tstep_kernel_core
////////////////////////////////////////////////////////////////////////////////
inline REAL
p2p_tstep_kernel_core(REAL iomega,
                      const REAL4 rmi, const REAL4 vei,
                      const REAL4 rmj, const REAL4 vej,
                      const REAL eta)
{
    REAL4 r;
    r.x = rmi.x - rmj.x;                                             // 1 FLOPs
    r.y = rmi.y - rmj.y;                                             // 1 FLOPs
    r.z = rmi.z - rmj.z;                                             // 1 FLOPs
    r.w = rmi.w + rmj.w;                                             // 1 FLOPs
    REAL4 v;
    v.x = vei.x - vej.x;                                             // 1 FLOPs
    v.y = vei.y - vej.y;                                             // 1 FLOPs
    v.z = vei.z - vej.z;                                             // 1 FLOPs
    v.w = vei.w + vej.w;                                             // 1 FLOPs
    REAL r2 = r.x * r.x + r.y * r.y + r.z * r.z;                     // 5 FLOPs
    REAL rv = r.x * v.x + r.y * v.y + r.z * v.z;                     // 5 FLOPs
    REAL v2 = v.x * v.x + v.y * v.y + v.z * v.z;                     // 5 FLOPs

    REAL2 ret = smoothed_inv_r2r3(r2, v.w);                          // 4 FLOPs
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
// Total flop count: 51


#ifdef __OPENCL_VERSION__
//
// OpenCL implementation
////////////////////////////////////////////////////////////////////////////////
inline REAL
p2p_accum_tstep(REAL iOmega,
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
    uint j;
    for (j = j_begin; j < j_end; ++j) {
        REAL8 jData = (REAL8){sharedJObj_x[j],
                              sharedJObj_y[j],
                              sharedJObj_z[j],
                              sharedJObj_mass[j],
                              sharedJObj_vx[j],
                              sharedJObj_vy[j],
                              sharedJObj_vz[j],
                              sharedJObj_eps2[j]};
        iOmega = p2p_tstep_kernel_core(iOmega,
                                       iData.lo, iData.hi,
                                       jData.lo, jData.hi,
                                       eta);
    }
    return iOmega;
}


inline REAL
p2p_tstep_kernel_main_loop(const REAL8 iData,
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

    uint tile;
    uint numTiles = (nj - 1)/lsize + 1;
    for (tile = 0; tile < numTiles; ++tile) {
        uint nb = min(lsize, (nj - (tile * lsize)));

        event_t e[8];
        e[0] = async_work_group_copy(sharedJObj_x, jobj_x + tile * lsize, nb, 0);
        e[1] = async_work_group_copy(sharedJObj_y, jobj_y + tile * lsize, nb, 0);
        e[2] = async_work_group_copy(sharedJObj_z, jobj_z + tile * lsize, nb, 0);
        e[3] = async_work_group_copy(sharedJObj_mass, jobj_mass + tile * lsize, nb, 0);
        e[4] = async_work_group_copy(sharedJObj_vx, jobj_vx + tile * lsize, nb, 0);
        e[5] = async_work_group_copy(sharedJObj_vy, jobj_vy + tile * lsize, nb, 0);
        e[6] = async_work_group_copy(sharedJObj_vz, jobj_vz + tile * lsize, nb, 0);
        e[7] = async_work_group_copy(sharedJObj_eps2, jobj_eps2 + tile * lsize, nb, 0);
        wait_group_events(8, e);

        uint j = 0;
        uint j_max = (nb > (JUNROLL - 1)) ? (nb - (JUNROLL - 1)):(0);
        for (; j < j_max; j += JUNROLL) {
            iOmega = p2p_accum_tstep(iOmega, iData,
                                     eta, j, j + JUNROLL,
                                     sharedJObj_x, sharedJObj_y, sharedJObj_z, sharedJObj_mass,
                                     sharedJObj_vx, sharedJObj_vy, sharedJObj_vz, sharedJObj_eps2);
        }
        iOmega = p2p_accum_tstep(iOmega, iData,
                                 eta, j, nb,
                                 sharedJObj_x, sharedJObj_y, sharedJObj_z, sharedJObj_mass,
                                 sharedJObj_vx, sharedJObj_vy, sharedJObj_vz, sharedJObj_eps2);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return iOmega;
}


__kernel void p2p_tstep_kernel(const uint ni,
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

    REAL iomega = p2p_tstep_kernel_main_loop(iData,
                                             nj,
                                             jobj_x,
                                             jobj_y,
                                             jobj_z,
                                             jobj_mass,
                                             jobj_vx,
                                             jobj_vy,
                                             jobj_vz,
                                             jobj_eps2,
                                             eta,
                                             sharedJObj_x,
                                             sharedJObj_y,
                                             sharedJObj_z,
                                             sharedJObj_mass,
                                             sharedJObj_vx,
                                             sharedJObj_vy,
                                             sharedJObj_vz,
                                             sharedJObj_eps2);
//    itstep[i] = 2 * eta / iomega;
    itstep[i] = 2 * eta / sqrt(iomega);
}

#else
//
// C implementation
////////////////////////////////////////////////////////////////////////////////
static PyObject *
_p2p_tstep_kernel(PyObject *_args)
{
    unsigned int ni, nj;
    REAL eta;
    // i-objs
    PyObject *iobj_x = NULL;
    PyObject *iobj_y = NULL;
    PyObject *iobj_z = NULL;
    PyObject *iobj_mass = NULL;
    PyObject *iobj_vx = NULL;
    PyObject *iobj_vy = NULL;
    PyObject *iobj_vz = NULL;
    PyObject *iobj_eps2 = NULL;
    // j-objs
    PyObject *jobj_x = NULL;
    PyObject *jobj_y = NULL;
    PyObject *jobj_z = NULL;
    PyObject *jobj_mass = NULL;
    PyObject *jobj_vx = NULL;
    PyObject *jobj_vy = NULL;
    PyObject *jobj_vz = NULL;
    PyObject *jobj_eps2 = NULL;
    PyObject *_output = NULL;

    int typenum;
    char *fmt = NULL;
    if (sizeof(REAL) == sizeof(double)) {
        fmt = "IOOOOOOOOIOOOOOOOOdO!";
        typenum = NPY_FLOAT64;
    } else if (sizeof(REAL) == sizeof(float)) {
        fmt = "IOOOOOOOOIOOOOOOOOfO!";
        typenum = NPY_FLOAT32;
    }

    if (!PyArg_ParseTuple(_args, fmt, &ni,
                                      &iobj_x, &iobj_y, &iobj_z, &iobj_mass,
                                      &iobj_vx, &iobj_vy, &iobj_vz, &iobj_eps2,
                                      &nj,
                                      &jobj_x, &jobj_y, &jobj_z, &jobj_mass,
                                      &jobj_vx, &jobj_vy, &jobj_vz, &jobj_eps2,
                                      &eta,
                                      &PyArray_Type, &_output))
        return NULL;

    // i-data
    PyObject *iobj_x_arr = PyArray_FROM_OTF(iobj_x, typenum, NPY_IN_ARRAY);
    REAL *iobj_x_ptr = (REAL *)PyArray_DATA(iobj_x_arr);

    PyObject *iobj_y_arr = PyArray_FROM_OTF(iobj_y, typenum, NPY_IN_ARRAY);
    REAL *iobj_y_ptr = (REAL *)PyArray_DATA(iobj_y_arr);

    PyObject *iobj_z_arr = PyArray_FROM_OTF(iobj_z, typenum, NPY_IN_ARRAY);
    REAL *iobj_z_ptr = (REAL *)PyArray_DATA(iobj_z_arr);

    PyObject *iobj_mass_arr = PyArray_FROM_OTF(iobj_mass, typenum, NPY_IN_ARRAY);
    REAL *iobj_mass_ptr = (REAL *)PyArray_DATA(iobj_mass_arr);

    PyObject *iobj_vx_arr = PyArray_FROM_OTF(iobj_vx, typenum, NPY_IN_ARRAY);
    REAL *iobj_vx_ptr = (REAL *)PyArray_DATA(iobj_vx_arr);

    PyObject *iobj_vy_arr = PyArray_FROM_OTF(iobj_vy, typenum, NPY_IN_ARRAY);
    REAL *iobj_vy_ptr = (REAL *)PyArray_DATA(iobj_vy_arr);

    PyObject *iobj_vz_arr = PyArray_FROM_OTF(iobj_vz, typenum, NPY_IN_ARRAY);
    REAL *iobj_vz_ptr = (REAL *)PyArray_DATA(iobj_vz_arr);

    PyObject *iobj_eps2_arr = PyArray_FROM_OTF(iobj_eps2, typenum, NPY_IN_ARRAY);
    REAL *iobj_eps2_ptr = (REAL *)PyArray_DATA(iobj_eps2_arr);

    // j-data
    PyObject *jobj_x_arr = PyArray_FROM_OTF(jobj_x, typenum, NPY_IN_ARRAY);
    REAL *jobj_x_ptr = (REAL *)PyArray_DATA(jobj_x_arr);

    PyObject *jobj_y_arr = PyArray_FROM_OTF(jobj_y, typenum, NPY_IN_ARRAY);
    REAL *jobj_y_ptr = (REAL *)PyArray_DATA(jobj_y_arr);

    PyObject *jobj_z_arr = PyArray_FROM_OTF(jobj_z, typenum, NPY_IN_ARRAY);
    REAL *jobj_z_ptr = (REAL *)PyArray_DATA(jobj_z_arr);

    PyObject *jobj_mass_arr = PyArray_FROM_OTF(jobj_mass, typenum, NPY_IN_ARRAY);
    REAL *jobj_mass_ptr = (REAL *)PyArray_DATA(jobj_mass_arr);

    PyObject *jobj_vx_arr = PyArray_FROM_OTF(jobj_vx, typenum, NPY_IN_ARRAY);
    REAL *jobj_vx_ptr = (REAL *)PyArray_DATA(jobj_vx_arr);

    PyObject *jobj_vy_arr = PyArray_FROM_OTF(jobj_vy, typenum, NPY_IN_ARRAY);
    REAL *jobj_vy_ptr = (REAL *)PyArray_DATA(jobj_vy_arr);

    PyObject *jobj_vz_arr = PyArray_FROM_OTF(jobj_vz, typenum, NPY_IN_ARRAY);
    REAL *jobj_vz_ptr = (REAL *)PyArray_DATA(jobj_vz_arr);

    PyObject *jobj_eps2_arr = PyArray_FROM_OTF(jobj_eps2, typenum, NPY_IN_ARRAY);
    REAL *jobj_eps2_ptr = (REAL *)PyArray_DATA(jobj_eps2_arr);

    // output-array
    PyObject *ret = PyArray_FROM_OTF(_output, typenum, NPY_INOUT_ARRAY);
    REAL *ret_ptr = (REAL *)PyArray_DATA(ret);

    // main calculation
    unsigned int i, j;
    for (i = 0; i < ni; ++i) {
        REAL4 rmi = {iobj_x_ptr[i], iobj_y_ptr[i],
                     iobj_z_ptr[i], iobj_mass_ptr[i]};
        REAL4 vei = {iobj_vx_ptr[i], iobj_vy_ptr[i],
                     iobj_vz_ptr[i], iobj_eps2_ptr[i]};
        REAL iomega = (REAL)0;
        for (j = 0; j < nj; ++j) {
            REAL4 rmj = {jobj_x_ptr[j], jobj_y_ptr[j],
                         jobj_z_ptr[j], jobj_mass_ptr[j]};
            REAL4 vej = {jobj_vx_ptr[j], jobj_vy_ptr[j],
                         jobj_vz_ptr[j], jobj_eps2_ptr[j]};
            iomega = p2p_tstep_kernel_core(iomega, rmi, vei, rmj, vej, eta);
        }
//        ret_ptr[i] = 2 * eta / iomega;
        ret_ptr[i] = 2 * eta / sqrt(iomega);
    }

    // Decrement the reference counts for i-objects
    Py_DECREF(iobj_x_arr);
    Py_DECREF(iobj_y_arr);
    Py_DECREF(iobj_z_arr);
    Py_DECREF(iobj_mass_arr);
    Py_DECREF(iobj_vx_arr);
    Py_DECREF(iobj_vy_arr);
    Py_DECREF(iobj_vz_arr);
    Py_DECREF(iobj_eps2_arr);

    // Decrement the reference counts for j-objects
    Py_DECREF(jobj_x_arr);
    Py_DECREF(jobj_y_arr);
    Py_DECREF(jobj_z_arr);
    Py_DECREF(jobj_mass_arr);
    Py_DECREF(jobj_vx_arr);
    Py_DECREF(jobj_vy_arr);
    Py_DECREF(jobj_vz_arr);
    Py_DECREF(jobj_eps2_arr);

    // Decrement the reference counts for ret-objects
    Py_DECREF(ret);

    Py_INCREF(Py_None);
    return Py_None;
}

#endif  // __OPENCL_VERSION__


#endif  // P2P_TSTEP_KERNEL_H

