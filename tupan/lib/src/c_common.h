#ifndef __C_COMMON_H__
#define __C_COMMON_H__

#include <stdio.h>
#include <tgmath.h>

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

#define paster(x,y) x##y
#define concat(x,y) paster(x,y)
#define vec(x) concat(x, 1)

#define int_tn vec(int_t)
#define uint_tn vec(uint_t)
#define real_tn vec(real_t)

#define rsqrt(x) (1 / sqrt(x))
#define select(a, b, c) ((c) ? (b):(a))

#endif // __C_COMMON_H__
