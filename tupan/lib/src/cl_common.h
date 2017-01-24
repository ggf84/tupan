#ifndef __CL_COMMON_H__
#define __CL_COMMON_H__

#if !defined(CL_VERSION_1_2)
	#if defined(cl_khr_fp64)
		#pragma OPENCL EXTENSION cl_khr_fp64 : enable
	#else
		#error "Missing double precision extension"
	#endif
#endif

#define DEFINE_TYPE(TYPEA, TYPEB)	\
	typedef TYPEA TYPEB;			\
	typedef TYPEA TYPEB##1;			\
	typedef TYPEA##2 TYPEB##2;		\
	typedef TYPEA##4 TYPEB##4;		\
	typedef TYPEA##8 TYPEB##8;		\
	typedef TYPEA##16 TYPEB##16;

#ifdef CONFIG_USE_DOUBLE
	DEFINE_TYPE(long, int_t)
	DEFINE_TYPE(ulong, uint_t)
	DEFINE_TYPE(double, real_t)
#else
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

#endif	// __CL_COMMON_H__
