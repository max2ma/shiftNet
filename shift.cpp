#include "para.h"
#include "shift.h"
#include "norm_relu.h"
using namespace para;


void shift(DataType tensor[D][D][M],
		DataType act[nD][nD][N]){
#pragma HLS INTERFACE m_axi depth=16384 port=tensor bundle=gmem
#pragma HLS INTERFACE m_axi depth=65536 port=act bundle=gmem
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE s_axilite port=tensor bundle=control
#pragma HLS INTERFACE s_axilite port=act bundle=control

		const int Dx[M]={
#include "dx"
		};
		const int Dy[M]={
#include "dy"
		};
		const DataType p[M][N]={
#include "p"
		};

		const DataType ave[N]={
#include "ave"
		};
		const DataType std[N]={
#include "std"
		};

#pragma HLS DATAFLOW
		DataType fmap[nD][nD][N];

		_shift_s<DataType, D, M, N, S>(tensor, fmap, Dx, Dy, p);
		_norm_relu<DataType, nD, N>(fmap,ave,std, act);
}



