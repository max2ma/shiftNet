#pragma once
#include "ap_fixed.h"
#include "hls_half.h"

#ifdef FIXED
typedef ap_fixed<32, 10> DataType;
#elif defined FLOAT
typedef float DataType;
#elif defined DOUBLE
typedef double DataType;
#else
typedef half DataType;
#endif
namespace para{
	const int D = 32;
	const int C = 3;
	const int N = 10;
}
