#include <cstring>
#include "hls_stream.h"
#include "para.h"
#include "shift.h"
using namespace para;

#define MUL

#ifndef MUL
void shift(hls::stream<DataType> & tensor,
		hls::stream<DataType> & act){
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE axis register both port=tensor
#pragma HLS INTERFACE axis register both port=act

		const int D0[C * E]={
#include "d_0"
		};
		const DataType p0[C][C * E]={
#include "p0_0"
		};
		const DataType p1[C * E][N]={
#include "p1_0"
		};

#pragma HLS DATAFLOW
		SingleChan::_shift<DataType, D, sS, cS, C, E, N>(tensor, act, D0 ,p0,p1);
}
#else

void shift(hls::stream<DataType>  tensor[C],
		hls::stream<DataType> act[N]){
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE axis register both port=tensor
#pragma HLS INTERFACE axis register both port=act

		const int D0[C * E]={
#include "d_0"
		};
		const DataType p0[C][C * E]={
#include "p0_0"
		};
		const DataType p1[C * E][N]={
#include "p1_0"
		};

#pragma HLS DATAFLOW
		//SingleChan::_shift<DataType, D, sS, cS, C, E, N>(tensor, act, D0 ,p0,p1);
		MulChan::_shift<DataType, D, sS, cS, C, E, N>(tensor, act, D0 ,p0,p1);
}


#endif



