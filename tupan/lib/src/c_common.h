#ifndef __C_COMMON_H__
#define __C_COMMON_H__

#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

#define DEFINE_TYPE(TYPEA, TYPEB)	\
	typedef TYPEA TYPEB;			\
	typedef TYPEA TYPEB##1;			\
	typedef TYPEA TYPEB##2;			\
	typedef TYPEA TYPEB##4;			\
	typedef TYPEA TYPEB##8;			\
	typedef TYPEA TYPEB##16;

#ifdef CONFIG_USE_DOUBLE
	DEFINE_TYPE(long, int_t)
	DEFINE_TYPE(unsigned long, uint_t)
	DEFINE_TYPE(double, real_t)
#else
	DEFINE_TYPE(int, int_t)
	DEFINE_TYPE(unsigned int, uint_t)
	DEFINE_TYPE(float, real_t)
#endif

extern "C" {
	#include "type_defs.h"
	#include "libtupan.h"
}

#define constant const
#define rsqrt(x) (1 / sqrt(x))
#define select(a, b, c) ((c) ? (b):(a))

#endif // __C_COMMON_H__
