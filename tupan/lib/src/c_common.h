#ifndef __C_COMMON_H__
#define __C_COMMON_H__

#include <stdio.h>
#include <tgmath.h>

#ifdef DOUBLE
    typedef long INT;
    typedef long INT2;
    typedef long INT4;
    typedef long INT8;
    typedef long INT16;

    typedef unsigned long UINT;
    typedef unsigned long UINT2;
    typedef unsigned long UINT4;
    typedef unsigned long UINT8;
    typedef unsigned long UINT16;

    typedef double REAL;
    typedef double REAL2;
    typedef double REAL4;
    typedef double REAL8;
    typedef double REAL16;
#else
    typedef int INT;
    typedef int INT2;
    typedef int INT4;
    typedef int INT8;
    typedef int INT16;

    typedef unsigned int UINT;
    typedef unsigned int UINT2;
    typedef unsigned int UINT4;
    typedef unsigned int UINT8;
    typedef unsigned int UINT16;

    typedef float REAL;
    typedef float REAL2;
    typedef float REAL4;
    typedef float REAL8;
    typedef float REAL16;
#endif

#define paster(x,y) x##y
#define concat(x,y) paster(x,y)
#define vec(x) concat(x, 1)

#define INT1 INT
#define UINT1 UINT
#define REAL1 REAL

#define INTn vec(INT)
#define UINTn vec(UINT)
#define REALn vec(REAL)

#define min fmin
#define max fmax
#define select(a, b, c) ((c) ? (b):(a))

#include "libtupan.h"

#endif // __C_COMMON_H__
