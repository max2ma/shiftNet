#pragma once

#ifdef FIXED
#include "ap_fixed.h"
typedef ap_fixed<32, 10> DataType;
#elif defined FLOAT
typedef float DataType;
#elif defined DOUBLE
typedef double DataType;
#else
#include "hls_half.h"
typedef half DataType;
#endif

