#ifndef P2P_ACC_KERNEL_H
#define P2P_ACC_KERNEL_H

#include"common.h"
#include"smoothing.h"

//
// p2p_acc_kernel_core
////////////////////////////////////////////////////////////////////////////////
inline REAL3
p2p_acc_kernel_core(REAL3 acc,
                    const REAL4 rmi, const REAL4 vei,
                    const REAL4 rmj, const REAL4 vej)
{
    REAL3 r;
    r.x = rmi.x - rmj.x;                                             // 1 FLOPs
    r.y = rmi.y - rmj.y;                                             // 1 FLOPs
    r.z = rmi.z - rmj.z;                                             // 1 FLOPs
    REAL r2 = r.x * r.x + r.y * r.y + r.z * r.z;                     // 5 FLOPs
    REAL inv_r3 = smoothed_inv_r3(r2, vei.w + vej.w);                // 5 FLOPs

    inv_r3 *= rmj.w;                                                 // 1 FLOPs

    acc.x -= inv_r3 * r.x;                                           // 2 FLOPs
    acc.y -= inv_r3 * r.y;                                           // 2 FLOPs
    acc.z -= inv_r3 * r.z;                                           // 2 FLOPs
    return acc;
}
// Total flop count: 20


#ifdef __OPENCL_VERSION__
//
// OpenCL implementation
////////////////////////////////////////////////////////////////////////////////
inline REAL3
p2p_accum_acc(REAL3 iAcc,
              const REAL8 iData,
              uint j_begin,
              uint j_end,
//              __local REAL8 *sharedJData
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
//        REAL8 jData = sharedJData[j];
        REAL8 jData = (REAL8){sharedJObj_x[j], sharedJObj_y[j], sharedJObj_z[j], sharedJObj_mass[j], sharedJObj_vx[j], sharedJObj_vy[j], sharedJObj_vz[j], sharedJObj_eps2[j]};
        iAcc = p2p_acc_kernel_core(iAcc,
                                   iData.lo, iData.hi,
                                   jData.lo, jData.hi);
    }
    return iAcc;
}


inline REAL3
p2p_acc_kernel_main_loop(//const REAL8 iData,
                         const REAL iObj_x,
                         const REAL iObj_y,
                         const REAL iObj_z,
                         const REAL iObj_mass,
                         const REAL iObj_vx,
                         const REAL iObj_vy,
                         const REAL iObj_vz,
                         const REAL iObj_eps2,
                         const uint nj,
//                         __global const REAL8 *jdata,
                         __global const REAL *jobj_x,
                         __global const REAL *jobj_y,
                         __global const REAL *jobj_z,
                         __global const REAL *jobj_mass,
                         __global const REAL *jobj_vx,
                         __global const REAL *jobj_vy,
                         __global const REAL *jobj_vz,
                         __global const REAL *jobj_eps2,
//                         __local REAL8 *sharedJData
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

    const REAL8 iData = (REAL8){iObj_x, iObj_y, iObj_z, iObj_mass, iObj_vx, iObj_vy, iObj_vz, iObj_eps2};

    REAL3 iAcc = (REAL3){0, 0, 0};

    uint tile;
    uint numTiles = (nj - 1)/lsize + 1;
    for (tile = 0; tile < numTiles; ++tile) {
        uint nb = min(lsize, (nj - (tile * lsize)));

//        event_t e[1];
//        e[0] = async_work_group_copy(sharedJData, jdata + tile * lsize, nb, 0);
//        wait_group_events(1, e);

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
//            iAcc = p2p_accum_acc(iAcc, iData,
//                                 j, j + JUNROLL,
//                                 sharedJData);
            iAcc = p2p_accum_acc(iAcc, iData,
                                 j, j + JUNROLL,
                                 sharedJObj_x, sharedJObj_y, sharedJObj_z, sharedJObj_mass,
                                 sharedJObj_vx, sharedJObj_vy, sharedJObj_vz, sharedJObj_eps2);
        }
//        iAcc = p2p_accum_acc(iAcc, iData,
//                             j, nb,
//                             sharedJData);
        iAcc = p2p_accum_acc(iAcc, iData,
                             j, nb,
                             sharedJObj_x, sharedJObj_y, sharedJObj_z, sharedJObj_mass,
                             sharedJObj_vx, sharedJObj_vy, sharedJObj_vz, sharedJObj_eps2);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return iAcc;
}


__kernel void p2p_acc_kernel(const uint ni,
//                             __global const REAL8 *idata,
                             __global const REAL *iobj_x,
                             __global const REAL *iobj_y,
                             __global const REAL *iobj_z,
                             __global const REAL *iobj_mass,
                             __global const REAL *iobj_vx,
                             __global const REAL *iobj_vy,
                             __global const REAL *iobj_vz,
                             __global const REAL *iobj_eps2,
                             const uint nj,
//                             __global const REAL8 *jdata,
                             __global const REAL *jobj_x,
                             __global const REAL *jobj_y,
                             __global const REAL *jobj_z,
                             __global const REAL *jobj_mass,
                             __global const REAL *jobj_vx,
                             __global const REAL *jobj_vy,
                             __global const REAL *jobj_vz,
                             __global const REAL *jobj_eps2,
                             __global REAL3 *iacc,
//                             __local REAL8 *sharedJData
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
//    iacc[i] = p2p_acc_kernel_main_loop(idata[i],
//                                       nj, jdata,
//                                       sharedJData);
    iacc[i] = p2p_acc_kernel_main_loop(iobj_x[i], iobj_y[i], iobj_z[i], iobj_mass[i],
                                       iobj_vx[i], iobj_vy[i], iobj_vz[i], iobj_eps2[i],
                                       nj,
                                       jobj_x, jobj_y, jobj_z, jobj_mass,
                                       jobj_vx, jobj_vy, jobj_vz, jobj_eps2,
                                       sharedJObj_x, sharedJObj_y, sharedJObj_z, sharedJObj_mass,
                                       sharedJObj_vx, sharedJObj_vy, sharedJObj_vz, sharedJObj_eps2);
}

#else
//
// C implementation
////////////////////////////////////////////////////////////////////////////////
static PyObject *
_p2p_acc_kernel(PyObject *_args)
{
    unsigned int ni, nj;
//    PyObject *_idata = NULL;
    PyObject *_iobj_x = NULL;
    PyObject *_iobj_y = NULL;
    PyObject *_iobj_z = NULL;
    PyObject *_iobj_mass = NULL;
    PyObject *_iobj_vx = NULL;
    PyObject *_iobj_vy = NULL;
    PyObject *_iobj_vz = NULL;
    PyObject *_iobj_eps2 = NULL;
//    PyObject *_jdata = NULL;
    PyObject *_jobj_x = NULL;
    PyObject *_jobj_y = NULL;
    PyObject *_jobj_z = NULL;
    PyObject *_jobj_mass = NULL;
    PyObject *_jobj_vx = NULL;
    PyObject *_jobj_vy = NULL;
    PyObject *_jobj_vz = NULL;
    PyObject *_jobj_eps2 = NULL;
    PyObject *_output = NULL;

    int typenum;
    char *fmt = NULL;
    if (sizeof(REAL) == sizeof(double)) {
//        fmt = "IOIOO!";
        fmt = "IOOOOOOOOIOOOOOOOOO!";
        typenum = NPY_FLOAT64;
    } else if (sizeof(REAL) == sizeof(float)) {
//        fmt = "IOIOO!";
        fmt = "IOOOOOOOOIOOOOOOOOO!";
        typenum = NPY_FLOAT32;
    }

//    if (!PyArg_ParseTuple(_args, fmt, &ni, &_idata,
//                                      &nj, &_jdata,
//                                      &PyArray_Type, &_output))
//        return NULL;

    if (!PyArg_ParseTuple(_args, fmt, &ni,
                                      &_iobj_x, &_iobj_y, &_iobj_z, &_iobj_mass,
                                      &_iobj_vx, &_iobj_vy, &_iobj_vz, &_iobj_eps2,
                                      &nj,
                                      &_jobj_x, &_jobj_y, &_jobj_z, &_jobj_mass,
                                      &_jobj_vx, &_jobj_vy, &_jobj_vz, &_jobj_eps2,
                                      &PyArray_Type, &_output))
        return NULL;

    // i-data
//    PyObject *_idata_arr = PyArray_FROM_OTF(_idata, typenum, NPY_IN_ARRAY);
//    REAL *idata_ptr = (REAL *)PyArray_DATA(_idata_arr);

    PyObject *_iobj_x_arr = PyArray_FROM_OTF(_iobj_x, typenum, NPY_IN_ARRAY);
    REAL *iobj_x_ptr = (REAL *)PyArray_DATA(_iobj_x_arr);

    PyObject *_iobj_y_arr = PyArray_FROM_OTF(_iobj_y, typenum, NPY_IN_ARRAY);
    REAL *iobj_y_ptr = (REAL *)PyArray_DATA(_iobj_y_arr);

    PyObject *_iobj_z_arr = PyArray_FROM_OTF(_iobj_z, typenum, NPY_IN_ARRAY);
    REAL *iobj_z_ptr = (REAL *)PyArray_DATA(_iobj_z_arr);

    PyObject *_iobj_mass_arr = PyArray_FROM_OTF(_iobj_mass, typenum, NPY_IN_ARRAY);
    REAL *iobj_mass_ptr = (REAL *)PyArray_DATA(_iobj_mass_arr);

    PyObject *_iobj_vx_arr = PyArray_FROM_OTF(_iobj_vx, typenum, NPY_IN_ARRAY);
    REAL *iobj_vx_ptr = (REAL *)PyArray_DATA(_iobj_vx_arr);

    PyObject *_iobj_vy_arr = PyArray_FROM_OTF(_iobj_vy, typenum, NPY_IN_ARRAY);
    REAL *iobj_vy_ptr = (REAL *)PyArray_DATA(_iobj_vy_arr);

    PyObject *_iobj_vz_arr = PyArray_FROM_OTF(_iobj_vz, typenum, NPY_IN_ARRAY);
    REAL *iobj_vz_ptr = (REAL *)PyArray_DATA(_iobj_vz_arr);

    PyObject *_iobj_eps2_arr = PyArray_FROM_OTF(_iobj_eps2, typenum, NPY_IN_ARRAY);
    REAL *iobj_eps2_ptr = (REAL *)PyArray_DATA(_iobj_eps2_arr);

    // j-data
//    PyObject *_jdata_arr = PyArray_FROM_OTF(_jdata, typenum, NPY_IN_ARRAY);
//    REAL *jdata_ptr = (REAL *)PyArray_DATA(_jdata_arr);

    PyObject *_jobj_x_arr = PyArray_FROM_OTF(_jobj_x, typenum, NPY_IN_ARRAY);
    REAL *jobj_x_ptr = (REAL *)PyArray_DATA(_jobj_x_arr);

    PyObject *_jobj_y_arr = PyArray_FROM_OTF(_jobj_y, typenum, NPY_IN_ARRAY);
    REAL *jobj_y_ptr = (REAL *)PyArray_DATA(_jobj_y_arr);

    PyObject *_jobj_z_arr = PyArray_FROM_OTF(_jobj_z, typenum, NPY_IN_ARRAY);
    REAL *jobj_z_ptr = (REAL *)PyArray_DATA(_jobj_z_arr);

    PyObject *_jobj_mass_arr = PyArray_FROM_OTF(_jobj_mass, typenum, NPY_IN_ARRAY);
    REAL *jobj_mass_ptr = (REAL *)PyArray_DATA(_jobj_mass_arr);

    PyObject *_jobj_vx_arr = PyArray_FROM_OTF(_jobj_vx, typenum, NPY_IN_ARRAY);
    REAL *jobj_vx_ptr = (REAL *)PyArray_DATA(_jobj_vx_arr);

    PyObject *_jobj_vy_arr = PyArray_FROM_OTF(_jobj_vy, typenum, NPY_IN_ARRAY);
    REAL *jobj_vy_ptr = (REAL *)PyArray_DATA(_jobj_vy_arr);

    PyObject *_jobj_vz_arr = PyArray_FROM_OTF(_jobj_vz, typenum, NPY_IN_ARRAY);
    REAL *jobj_vz_ptr = (REAL *)PyArray_DATA(_jobj_vz_arr);

    PyObject *_jobj_eps2_arr = PyArray_FROM_OTF(_jobj_eps2, typenum, NPY_IN_ARRAY);
    REAL *jobj_eps2_ptr = (REAL *)PyArray_DATA(_jobj_eps2_arr);

    // output-array
    PyObject *ret = PyArray_FROM_OTF(_output, typenum, NPY_INOUT_ARRAY);
    REAL *ret_ptr = (REAL *)PyArray_DATA(ret);
    int k = PyArray_DIM((PyArrayObject *)ret, 1);

    // main calculation
    unsigned int i, j;
    for (i = 0; i < ni; ++i) {
//        unsigned int i8 = 8*i;
//        REAL4 rmi = {idata_ptr[i8  ], idata_ptr[i8+1],
//                     idata_ptr[i8+2], idata_ptr[i8+3]};
//        REAL4 vei = {idata_ptr[i8+4], idata_ptr[i8+5],
//                     idata_ptr[i8+6], idata_ptr[i8+7]};
        REAL4 rmi = {iobj_x_ptr[i], iobj_y_ptr[i],
                     iobj_z_ptr[i], iobj_mass_ptr[i]};
        REAL4 vei = {iobj_vx_ptr[i], iobj_vy_ptr[i],
                     iobj_vz_ptr[i], iobj_eps2_ptr[i]};
        REAL3 iacc = (REAL3){0, 0, 0};
        for (j = 0; j < nj; ++j) {
//            unsigned int j8 = 8*j;
//            REAL4 rmj = {jdata_ptr[j8  ], jdata_ptr[j8+1],
//                         jdata_ptr[j8+2], jdata_ptr[j8+3]};
//            REAL4 vej = {jdata_ptr[j8+4], jdata_ptr[j8+5],
//                         jdata_ptr[j8+6], jdata_ptr[j8+7]};
            REAL4 rmj = {jobj_x_ptr[j], jobj_y_ptr[j],
                         jobj_z_ptr[j], jobj_mass_ptr[j]};
            REAL4 vej = {jobj_vx_ptr[j], jobj_vy_ptr[j],
                         jobj_vz_ptr[j], jobj_eps2_ptr[j]};
            iacc = p2p_acc_kernel_core(iacc, rmi, vei, rmj, vej);
        }
        unsigned int ik = i * k;
        ret_ptr[ik  ] = iacc.x;
        ret_ptr[ik+1] = iacc.y;
        ret_ptr[ik+2] = iacc.z;
    }

    // Decrement the reference counts for i-objects
//    Py_DECREF(_idata_arr);
    Py_DECREF(_iobj_x_arr);
    Py_DECREF(_iobj_y_arr);
    Py_DECREF(_iobj_z_arr);
    Py_DECREF(_iobj_mass_arr);
    Py_DECREF(_iobj_vx_arr);
    Py_DECREF(_iobj_vy_arr);
    Py_DECREF(_iobj_vz_arr);
    Py_DECREF(_iobj_eps2_arr);

    // Decrement the reference counts for j-objects
//    Py_DECREF(_jdata_arr);
    Py_DECREF(_jobj_x_arr);
    Py_DECREF(_jobj_y_arr);
    Py_DECREF(_jobj_z_arr);
    Py_DECREF(_jobj_mass_arr);
    Py_DECREF(_jobj_vx_arr);
    Py_DECREF(_jobj_vy_arr);
    Py_DECREF(_jobj_vz_arr);
    Py_DECREF(_jobj_eps2_arr);

    // Decrement the reference counts for ret-objects
    Py_DECREF(ret);

    Py_INCREF(Py_None);
    return Py_None;
}

#endif  // __OPENCL_VERSION__


#endif  // P2P_ACC_KERNEL_H

