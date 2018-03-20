#pragma once
#include "ap_fixed.h"
#include "hls_half.h"


#define FIXED

#ifdef FIXED
typedef ap_fixed<16, 4> DataType;
#else
typedef float DataType;
#endif
namespace para{
	const int D = 8;
	const int M = 2;
	const int K = 4;
	const int N = 4;
	const int S = 1;
	const int nD = (D - 1)/S + 1;
}
