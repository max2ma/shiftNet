#pragma once

#define FIXED
#ifdef FIXED
#include "ap_fixed.h"
typedef ap_fixed<16, 6, AP_RND, AP_SAT> DataType;
#elif defined FLOAT
typedef float DataType;
#elif defined DOUBLE
typedef double DataType;
#else
#include "hls_half.h"
typedef half DataType;
#endif

