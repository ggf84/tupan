#ifndef __C_COMMON_H__
#define __C_COMMON_H__

#include <stdio.h>
#include <tgmath.h>

#define min fmin
#define max fmax

#ifdef DOUBLE
    typedef double REAL;
#else
    typedef float REAL;
#endif

#endif // __C_COMMON_H__
