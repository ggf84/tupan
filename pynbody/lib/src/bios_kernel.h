#ifndef BIOS_KERNEL_H
#define BIOS_KERNEL_H

#include <stdlib.h>
#include"common.h"
#include"smoothing.h"

/*
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * *  BIOS - BInary-based n-bOdy Solver  * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 */


inline REAL
get_phi(const REAL m,
        const REAL x,
        const REAL y,
        const REAL z,
        const REAL eps2)
{
    REAL r2 = x * x + y * y + z * z;                                 // 5 FLOPs
    REAL inv_r = phi_smooth(r2, eps2);                               // 3 FLOPs
    return m * inv_r;                                                // 1 FLOPs
}


inline REAL3
get_acc(const REAL m,
        const REAL x,
        const REAL y,
        const REAL z,
        const REAL eps2)
{
    REAL r2 = x * x + y * y + z * z;                                 // 5 FLOPs
    REAL inv_r3 = acc_smooth(r2, eps2);                              // 4 FLOPs
    REAL m_r3 = m * inv_r3;                                          // 1 FLOPs
    REAL3 a;
    a.x = -m_r3 * x;                                                 // 1 FLOPs
    a.y = -m_r3 * y;                                                 // 1 FLOPs
    a.z = -m_r3 * z;                                                 // 1 FLOPs
    return a;
}


inline REAL8
leapfrog(const REAL dt,
         const REAL4 r0,
         const REAL4 v0)
{
    REAL4 r = r0;
    REAL4 v = v0;

    r.x += v.x * dt/2;
    r.y += v.y * dt/2;
    r.z += v.z * dt/2;

    REAL3 a = get_acc(r.w, r.x, r.y, r.z, v.w);

    v.x += a.x * dt;
    v.y += a.y * dt;
    v.z += a.z * dt;

    r.x += v.x * dt/2;
    r.y += v.y * dt/2;
    r.z += v.z * dt/2;

    return (REAL8){r.x, r.y, r.z, r.w, v.x, v.y, v.z, v.w};
}


inline REAL8
TTL_core(const REAL dt,
         const REAL u0,
         const REAL k0,
         const REAL4 r0,
         const REAL4 v0,
         REAL *t)
{
    REAL4 r = r0;
    REAL4 v = v0;

    REAL w0 = 2 * u0;
    REAL tau0 = dt / w0;
    *t += tau0;
    r.x += v.x * tau0;
    r.y += v.y * tau0;
    r.z += v.z * tau0;

    REAL r2 = r.x * r.x + r.y * r.y + r.z * r.z;
    REAL inv_r2 = 1 / (r2 + v.w);
    inv_r2 = (r2 > 0) ? (inv_r2):(0);

    v.x -= r.x * inv_r2 * dt;
    v.y -= r.y * inv_r2 * dt;
    v.z -= r.z * inv_r2 * dt;
    REAL k1 = (v.x * v.x + v.y * v.y + v.z * v.z)/2;

    REAL u1 = u0 + (k1 - k0);
    REAL w1 = 2 * u1;
    REAL tau1 = dt / w1;
    *t += tau1;
    r.x += v.x * tau1;
    r.y += v.y * tau1;
    r.z += v.z * tau1;
    return (REAL8){r.x, r.y, r.z, r.w, v.x, v.y, v.z, v.w};
}


inline REAL8
TTL(const REAL dt,
    const REAL4 r0,
    const REAL4 v0)
{
    REAL4 r = r0;
    REAL4 v = v0;

    REAL k0 = (v.x * v.x + v.y * v.y + v.z * v.z)/2;
    REAL u0 = get_phi(r.w, r.x, r.y, r.z, v.w);

    REAL tau = u0 * dt;
    REAL t = 0;
    REAL8 rv_new = TTL_core(tau, u0, k0, r, v, &t);

    while (fabs(t-dt)/dt > 5.9604644775390625E-8) {
        tau *= sqrt(dt/t);
        t = 0;
        rv_new = TTL_core(tau, u0, k0, r, v, &t);
    }

    return rv_new;
}


inline REAL8
twobody_solver(const REAL dt,
               const REAL4 r0,
               const REAL4 v0)
{

//    return leapfrog(dt, r0, v0);
    return TTL(dt, r0, v0);
}


//
// bios_kernel_core
////////////////////////////////////////////////////////////////////////////////
inline REAL8
bios_kernel_core(REAL8 iposvel,
                 const REAL4 ri, const REAL4 vi,
                 const REAL4 rj, const REAL4 vj,
                 const REAL dt)
{
    REAL4 r;
    r.x = ri.x - rj.x;                                               // 1 FLOPs
    r.y = ri.y - rj.y;                                               // 1 FLOPs
    r.z = ri.z - rj.z;                                               // 1 FLOPs
    r.w = ri.w + rj.w;                                               // 1 FLOPs
    REAL4 v;
    v.x = vi.x - vj.x;                                               // 1 FLOPs
    v.y = vi.y - vj.y;                                               // 1 FLOPs
    v.z = vi.z - vj.z;                                               // 1 FLOPs
    v.w = vi.w + vj.w;                                               // 1 FLOPs
    REAL mu = (ri.w * rj.w) / r.w;                                   // 2 FLOPs
    if (r.x == 0 && r.y == 0 && r.z == 0) {
        return iposvel;
    }

    REAL8 rv_new = twobody_solver(dt, r, v);                         // ? FLOPS

    iposvel.s0 += mu * ((rv_new.s0 - r.x) - v.x * dt);               // 4 FLOPs
    iposvel.s1 += mu * ((rv_new.s1 - r.y) - v.y * dt);               // 4 FLOPs
    iposvel.s2 += mu * ((rv_new.s2 - r.z) - v.z * dt);               // 4 FLOPs
    iposvel.s3  = 0;
    iposvel.s4 += mu * (rv_new.s4 - v.x);                            // 2 FLOPs
    iposvel.s5 += mu * (rv_new.s5 - v.y);                            // 2 FLOPs
    iposvel.s6 += mu * (rv_new.s6 - v.z);                            // 2 FLOPs
    iposvel.s7  = 0;

    return iposvel;
}
// Total flop count: 28 + ?




#ifdef __OPENCL_VERSION__
//
// OpenCL implementation
////////////////////////////////////////////////////////////////////////////////

#else
//
// C implementation
////////////////////////////////////////////////////////////////////////////////
static PyObject *
_bios_kernel(PyObject *_args)
{
    unsigned int ni, nj;
    REAL dt;
    PyObject *_idata = NULL;
    PyObject *_jdata = NULL;

    int typenum;
    char *fmt = NULL;
    if (sizeof(REAL) == sizeof(double)) {
        fmt = "IOIOd";
        typenum = NPY_FLOAT64;
    } else if (sizeof(REAL) == sizeof(float)) {
        fmt = "IOIOf";
        typenum = NPY_FLOAT32;
    }

    if (!PyArg_ParseTuple(_args, fmt, &ni, &_idata,
                                      &nj, &_jdata,
                                      &dt))
        return NULL;

    // i-data
    PyObject *_idata_arr = PyArray_FROM_OTF(_idata, typenum, NPY_IN_ARRAY);
    REAL *idata_ptr = (REAL *)PyArray_DATA(_idata_arr);

    // j-data
    PyObject *_jdata_arr = PyArray_FROM_OTF(_jdata, typenum, NPY_IN_ARRAY);
    REAL *jdata_ptr = (REAL *)PyArray_DATA(_jdata_arr);

    // allocate a PyArrayObject to be returned
    npy_intp dims[2] = {ni, 8};
    PyArrayObject *ret = (PyArrayObject *)PyArray_EMPTY(2, dims, typenum, 0);
    REAL *ret_ptr = (REAL *)PyArray_DATA(ret);

    // main calculation
    unsigned int i, i8, j, j8;
    for (i = 0; i < ni; ++i) {
        i8 = 8*i;
        REAL8 iposvel = (REAL8){0, 0, 0, 0, 0, 0, 0, 0};
        REAL4 ri = {idata_ptr[i8  ], idata_ptr[i8+1],
                    idata_ptr[i8+2], idata_ptr[i8+3]};
        REAL4 vi = {idata_ptr[i8+4], idata_ptr[i8+5],
                    idata_ptr[i8+6], idata_ptr[i8+7]};
        for (j = 0; j < nj; ++j) {
            j8 = 8*j;
            REAL4 rj = {jdata_ptr[j8  ], jdata_ptr[j8+1],
                        jdata_ptr[j8+2], jdata_ptr[j8+3]};
            REAL4 vj = {jdata_ptr[j8+4], jdata_ptr[j8+5],
                        jdata_ptr[j8+6], jdata_ptr[j8+7]};
            iposvel = bios_kernel_core(iposvel, ri, vi, rj, vj, dt);
        }
        ret_ptr[i8  ] = iposvel.s0 / ri.w + vi.x * dt;
        ret_ptr[i8+1] = iposvel.s1 / ri.w + vi.y * dt;
        ret_ptr[i8+2] = iposvel.s2 / ri.w + vi.z * dt;
        ret_ptr[i8+3] = iposvel.s3;
        ret_ptr[i8+4] = iposvel.s4 / ri.w;
        ret_ptr[i8+5] = iposvel.s5 / ri.w;
        ret_ptr[i8+6] = iposvel.s6 / ri.w;
        ret_ptr[i8+7] = iposvel.s7;
    }

    // Decrement the reference counts for i-objects
    Py_DECREF(_idata_arr);

    // Decrement the reference counts for j-objects
    Py_DECREF(_jdata_arr);

    // returns a PyArrayObject
    return PyArray_Return(ret);
}

#endif  // __OPENCL_VERSION__


#endif  // BIOS_KERNEL_H

