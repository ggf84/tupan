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

#define paster(x,y) x##y
#define concat(x,y) paster(x,y)
#define vec(x) concat(x, VW)

#define int_tn vec(int_t)
#define uint_tn vec(uint_t)
#define real_tn vec(real_t)

typedef int_t1 int_t1xm[IUNROLL];
typedef uint_t1 uint_t1xm[IUNROLL];
typedef real_t1 real_t1xm[IUNROLL];

typedef int_tn int_tnxm[IUNROLL];
typedef uint_tn uint_tnxm[IUNROLL];
typedef real_tn real_tnxm[IUNROLL];

#define vload1(_offset, _ptr) *(_ptr+_offset)
#define vloadn concat(vload, IUNROLL)

#define vstore1(_src, _offset, _ptr) *(_ptr+_offset) = _src
#define vstoren concat(vstore, IUNROLL)

#define aload1(_offset, _ptr)	\
	*(_ptr+_offset)
#define aload2(_offset, _ptr)	\
	aload1(0, _ptr+2*_offset), aload1(1, _ptr+2*_offset)
#define aload4(_offset, _ptr)	\
	aload2(0, _ptr+4*_offset), aload2(1, _ptr+4*_offset)
#define aload8(_offset, _ptr)	\
	aload4(0, _ptr+8*_offset), aload4(1, _ptr+8*_offset)
#define aload16(_offset, _ptr)	\
	aload8(0, _ptr+16*_offset), aload8(1, _ptr+16*_offset)
#define aloadn(_offset, _ptr) {concat(aload, IUNROLL)(_offset, _ptr)}

#define astore1(_src, _offset, _ptr)	\
	*(_ptr+_offset) = *(_src)
#define astore2(_src, _offset, _ptr)	\
	astore1(_src, 0, _ptr+2*_offset), astore1(_src+1, 1, _ptr+2*_offset)
#define astore4(_src, _offset, _ptr)	\
	astore2(_src, 0, _ptr+4*_offset), astore2(_src+2, 1, _ptr+4*_offset)
#define astore8(_src, _offset, _ptr)	\
	astore4(_src, 0, _ptr+8*_offset), astore4(_src+4, 1, _ptr+8*_offset)
#define astore16(_src, _offset, _ptr)	\
	astore8(_src, 0, _ptr+16*_offset), astore8(_src+8, 1, _ptr+16*_offset)
#define astoren concat(astore, IUNROLL)

#endif	// __CL_COMMON_H__
