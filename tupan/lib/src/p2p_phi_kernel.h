#ifndef P2P_PHI_KERNEL_H
#define P2P_PHI_KERNEL_H

#include"common.h"
#include"smoothing.h"

//
// p2p_phi_kernel_core
////////////////////////////////////////////////////////////////////////////////
inline REAL
p2p_phi_kernel_core(REAL iphi,
                    const REAL4 irm, const REAL4 ive,
                    const REAL4 jrm, const REAL4 jve)
{
    REAL3 r;
    r.x = irm.x - jrm.x;                                             // 1 FLOPs
    r.y = irm.y - jrm.y;                                             // 1 FLOPs
    r.z = irm.z - jrm.z;                                             // 1 FLOPs
    REAL r2 = r.x * r.x + r.y * r.y + r.z * r.z;                     // 5 FLOPs
    REAL inv_r = smoothed_inv_r1(r2, ive.w + jve.w);                 // 4+1 FLOPs
    iphi -= jrm.w * inv_r;                                           // 2 FLOPs
    return iphi;
}
// Total flop count: 15


#ifdef __OPENCL_VERSION__
//
// OpenCL implementation
////////////////////////////////////////////////////////////////////////////////
inline REAL
p2p_accum_phi(REAL iphi,
              const REAL8 idata,
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
        iphi = p2p_phi_kernel_core(iphi,
                                   idata.lo, idata.hi,
                                   jdata.lo, jdata.hi);
    }
    return iphi;
}


inline REAL
p2p_phi_kernel_main_loop(const REAL8 idata,
                         const uint nj,
                         __global const REAL *inp_jrx,
                         __global const REAL *inp_jry,
                         __global const REAL *inp_jrz,
                         __global const REAL *inp_jmass,
                         __global const REAL *inp_jvx,
                         __global const REAL *inp_jvy,
                         __global const REAL *inp_jvz,
                         __global const REAL *inp_jeps2,
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

    REAL iphi = (REAL)0;

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
            iphi = p2p_accum_phi(iphi, idata,
                                 j, j + JUNROLL,
                                 shr_jrx, shr_jry, shr_jrz, shr_jmass,
                                 shr_jvx, shr_jvy, shr_jvz, shr_jeps2);
        }
        iphi = p2p_accum_phi(iphi, idata,
                             j, nb,
                             shr_jrx, shr_jry, shr_jrz, shr_jmass,
                             shr_jvx, shr_jvy, shr_jvz, shr_jeps2);

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return iphi;
}


__kernel void p2p_phi_kernel(const uint ni,
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
                             __global REAL *out_iphi,
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

    REAL iphi = p2p_phi_kernel_main_loop(idata,
                                         nj,
                                         inp_jrx, inp_jry, inp_jrz, inp_jmass,
                                         inp_jvx, inp_jvy, inp_jvz, inp_jeps2,
                                         shr_jrx, shr_jry, shr_jrz, shr_jmass,
                                         shr_jvx, shr_jvy, shr_jvz, shr_jeps2);
    out_iphi[i] = iphi;
}

#else
//
// C implementation
////////////////////////////////////////////////////////////////////////////////

inline void
main_p2p_phi_kernel(const unsigned int ni,
                    const REAL *irx, const REAL *iry, const REAL *irz, const REAL *imass,
                    const REAL *ivx, const REAL *ivy, const REAL *ivz, const REAL *ieps2,
                    const unsigned int nj,
                    const REAL *jrx, const REAL *jry, const REAL *jrz, const REAL *jmass,
                    const REAL *jvx, const REAL *jvy, const REAL *jvz, const REAL *jeps2,
                    REAL *iphi)
{
    unsigned int i, j;
    for (i = 0; i < ni; ++i) {
        REAL4 irm = {irx[i], iry[i], irz[i], imass[i]};
        REAL4 ive = {ivx[i], ivy[i], ivz[i], ieps2[i]};
        REAL ip = (REAL)0;
        for (j = 0; j < nj; ++j) {
            REAL4 jrm = {jrx[j], jry[j], jrz[j], jmass[j]};
            REAL4 jve = {jvx[j], jvy[j], jvz[j], jeps2[j]};
            ip = p2p_phi_kernel_core(ip, irm, ive, jrm, jve);
        }
        iphi[i] = ip;
    }
}


#ifndef __USE_CTYPES__
static PyObject *
p2p_phi_kernel(PyObject *_self, PyObject *_args)
{
    unsigned int ni, nj;
    // i-objs
    PyObject *inp_irx = NULL;
    PyObject *inp_iry = NULL;
    PyObject *inp_irz = NULL;
    PyObject *inp_imass = NULL;
    PyObject *inp_ivx = NULL;
    PyObject *inp_ivy = NULL;
    PyObject *inp_ivz = NULL;
    PyObject *inp_ieps2 = NULL;
    // j-objs
    PyObject *inp_jrx = NULL;
    PyObject *inp_jry = NULL;
    PyObject *inp_jrz = NULL;
    PyObject *inp_jmass = NULL;
    PyObject *inp_jvx = NULL;
    PyObject *inp_jvy = NULL;
    PyObject *inp_jvz = NULL;
    PyObject *inp_jeps2 = NULL;
    // out-objs
    PyObject *out_iphi = NULL;

    int typenum;
    char *fmt = NULL;
    if (sizeof(REAL) == sizeof(double)) {
        fmt = "IOOOOOOOOIOOOOOOOOO!";
        typenum = NPY_FLOAT64;
    } else if (sizeof(REAL) == sizeof(float)) {
        fmt = "IOOOOOOOOIOOOOOOOOO!";
        typenum = NPY_FLOAT32;
    }

    if (!PyArg_ParseTuple(_args, fmt, &ni,
                                      &inp_irx, &inp_iry, &inp_irz, &inp_imass,
                                      &inp_ivx, &inp_ivy, &inp_ivz, &inp_ieps2,
                                      &nj,
                                      &inp_jrx, &inp_jry, &inp_jrz, &inp_jmass,
                                      &inp_jvx, &inp_jvy, &inp_jvz, &inp_jeps2,
                                      &PyArray_Type, &out_iphi))
        return NULL;

    // i-objs
    PyObject *irx = PyArray_FROM_OTF(inp_irx, typenum, NPY_IN_ARRAY);
    REAL *irx_ptr = (REAL *)PyArray_DATA(irx);

    PyObject *iry = PyArray_FROM_OTF(inp_iry, typenum, NPY_IN_ARRAY);
    REAL *iry_ptr = (REAL *)PyArray_DATA(iry);

    PyObject *irz = PyArray_FROM_OTF(inp_irz, typenum, NPY_IN_ARRAY);
    REAL *irz_ptr = (REAL *)PyArray_DATA(irz);

    PyObject *imass = PyArray_FROM_OTF(inp_imass, typenum, NPY_IN_ARRAY);
    REAL *imass_ptr = (REAL *)PyArray_DATA(imass);

    PyObject *ivx = PyArray_FROM_OTF(inp_ivx, typenum, NPY_IN_ARRAY);
    REAL *ivx_ptr = (REAL *)PyArray_DATA(ivx);

    PyObject *ivy = PyArray_FROM_OTF(inp_ivy, typenum, NPY_IN_ARRAY);
    REAL *ivy_ptr = (REAL *)PyArray_DATA(ivy);

    PyObject *ivz = PyArray_FROM_OTF(inp_ivz, typenum, NPY_IN_ARRAY);
    REAL *ivz_ptr = (REAL *)PyArray_DATA(ivz);

    PyObject *ieps2 = PyArray_FROM_OTF(inp_ieps2, typenum, NPY_IN_ARRAY);
    REAL *ieps2_ptr = (REAL *)PyArray_DATA(ieps2);

    // j-objs
    PyObject *jrx = PyArray_FROM_OTF(inp_jrx, typenum, NPY_IN_ARRAY);
    REAL *jrx_ptr = (REAL *)PyArray_DATA(jrx);

    PyObject *jry = PyArray_FROM_OTF(inp_jry, typenum, NPY_IN_ARRAY);
    REAL *jry_ptr = (REAL *)PyArray_DATA(jry);

    PyObject *jrz = PyArray_FROM_OTF(inp_jrz, typenum, NPY_IN_ARRAY);
    REAL *jrz_ptr = (REAL *)PyArray_DATA(jrz);

    PyObject *jmass = PyArray_FROM_OTF(inp_jmass, typenum, NPY_IN_ARRAY);
    REAL *jmass_ptr = (REAL *)PyArray_DATA(jmass);

    PyObject *jvx = PyArray_FROM_OTF(inp_jvx, typenum, NPY_IN_ARRAY);
    REAL *jvx_ptr = (REAL *)PyArray_DATA(jvx);

    PyObject *jvy = PyArray_FROM_OTF(inp_jvy, typenum, NPY_IN_ARRAY);
    REAL *jvy_ptr = (REAL *)PyArray_DATA(jvy);

    PyObject *jvz = PyArray_FROM_OTF(inp_jvz, typenum, NPY_IN_ARRAY);
    REAL *jvz_ptr = (REAL *)PyArray_DATA(jvz);

    PyObject *jeps2 = PyArray_FROM_OTF(inp_jeps2, typenum, NPY_IN_ARRAY);
    REAL *jeps2_ptr = (REAL *)PyArray_DATA(jeps2);

    // out-objs
    PyObject *iphi = PyArray_FROM_OTF(out_iphi, typenum, NPY_INOUT_ARRAY);
    REAL *iphi_ptr = (REAL *)PyArray_DATA(iphi);

    // main calculation
    main_p2p_phi_kernel(ni,
                        irx_ptr, iry_ptr, irz_ptr, imass_ptr,
                        ivx_ptr, ivy_ptr, ivz_ptr, ieps2_ptr,
                        nj,
                        jrx_ptr, jry_ptr, jrz_ptr, jmass_ptr,
                        jvx_ptr, jvy_ptr, jvz_ptr, jeps2_ptr,
                        iphi_ptr);

    // Decrement the reference counts for auxiliary i-objs
    Py_DECREF(irx);
    Py_DECREF(iry);
    Py_DECREF(irz);
    Py_DECREF(imass);
    Py_DECREF(ivx);
    Py_DECREF(ivy);
    Py_DECREF(ivz);
    Py_DECREF(ieps2);

    // Decrement the reference counts for auxiliary j-objs
    Py_DECREF(jrx);
    Py_DECREF(jry);
    Py_DECREF(jrz);
    Py_DECREF(jmass);
    Py_DECREF(jvx);
    Py_DECREF(jvy);
    Py_DECREF(jvz);
    Py_DECREF(jeps2);

    // Decrement the reference counts for auxiliary out-objs
    Py_DECREF(iphi);

    // Returns None
    Py_INCREF(Py_None);
    return Py_None;
}
#endif  // __USE_CTYPES__

#endif  // __OPENCL_VERSION__


#endif  // P2P_PHI_KERNEL_H

