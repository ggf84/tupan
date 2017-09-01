#ifndef __TYPE_DEFS_H__
#define __TYPE_DEFS_H__

#define paster(x,y) x##y
#define concat(x,y) paster(x,y)
#define vec(x) concat(x, SIMD)

typedef vec(int_t) int_tn;
typedef vec(uint_t) uint_tn;
typedef vec(real_t) real_tn;

#define PI ((real_t)(3.14159265358979323846))
#define PI2 ((real_t)(9.86960440108935861883))
#define PI_2 ((real_t)(1.57079632679489661923))
#define TWOPI ((real_t)(6.28318530717958647693))
#define FOURPI ((real_t)(1.25663706143591729539e+1))
#define THREE_FOURPI ((real_t)(2.3873241463784300365e-1))

#ifdef CONFIG_USE_DOUBLE
	#define TOLERANCE exp2((real_t)(-42))
#else
	#define TOLERANCE exp2((real_t)(-16))
#endif
#define MAXITER 64

#endif	// __TYPE_DEFS_H__
