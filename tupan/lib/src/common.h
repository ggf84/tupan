#ifndef __COMMON_H__
#define __COMMON_H__

#define NDIM 3	// number of spatial dimensions
#define __ALIGNED__ __attribute__((aligned))

#ifdef CONFIG_USE_OPENCL
	#include "cl_common.h"
#else
	#include "c_common.h"
#endif

#endif // __COMMON_H__
