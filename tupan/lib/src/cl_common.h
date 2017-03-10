#ifndef __CL_COMMON_H__
#define __CL_COMMON_H__

#define DEFINE_TYPE(TYPEA, TYPEB)	\
	typedef TYPEA TYPEB;			\
	typedef TYPEA TYPEB##1;			\
	typedef TYPEA##2 TYPEB##2;		\
	typedef TYPEA##4 TYPEB##4;		\
	typedef TYPEA##8 TYPEB##8;		\
	typedef TYPEA##16 TYPEB##16;

#ifdef CONFIG_USE_DOUBLE
	#pragma OPENCL EXTENSION cl_khr_fp64 : enable
	#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
	#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
	DEFINE_TYPE(long, int_t)
	DEFINE_TYPE(ulong, uint_t)
	DEFINE_TYPE(double, real_t)
#else
	#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
	#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
	DEFINE_TYPE(int, int_t)
	DEFINE_TYPE(uint, uint_t)
	DEFINE_TYPE(float, real_t)
#endif

#include "type_defs.h"

#define shuff1(_x_)
#define shuff2(_x_) _x_ = _x_.s10
#define shuff4(_x_) _x_ = _x_.s1230
#define shuff8(_x_) _x_ = _x_.s12345670
#define shuff16(_x_) _x_ = _x_.s123456789abcdef0

#define shuff(_x_, SIMD) concat(shuff, SIMD)(_x_)
/*
// adapted from: https://streamcomputing.eu/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/
inline void
atomic_fadd(global real_t *addr, real_t value)
{
	union {
		real_t f;
		uint_t i;
	} prev, curr, next;
	curr.f = *((volatile global real_t *)addr);
	do {
		prev.f = curr.f;
		next.f = prev.f + value;
		curr.i = atom_cmpxchg((volatile global uint_t *)addr, prev.i, next.i);
	} while (curr.i != prev.i);
}
*/

// adapted from: https://devtalk.nvidia.com/default/topic/458062/atomicadd-float-float-atomicmul-float-float-/
inline void
atomic_fadd(global real_t* addr, real_t value)
{
	union {
		real_t f;
		uint_t i;
	} prev, curr;
    prev.f = value;
	do {
		curr.i = atom_xchg((volatile global uint_t *)addr, (uint_t)0);
		curr.f += prev.f;
		prev.i = atom_xchg((volatile global uint_t *)addr, curr.i);
	} while (prev.i != 0);
}

#endif	// __CL_COMMON_H__
