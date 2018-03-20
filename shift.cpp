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

	const int Dx1[M]={
	#include "dx1"
			};
			const int Dy1[M]={
	#include "dy1"
			};
			const DataType p1[M][K]={
	#include "p1"
			};

			const DataType ave1[K]={
	#include "ave1"
			};
			const DataType std1[K]={
	#include "std1"
			};

		DataType fmap1[D][D][K];
		DataType out1[D][D][K];
		_shift_n<DataType, D, M, K>(tensor, fmap1, Dx1, Dy1, p1);
		_norm_relu<DataType, D, K>(fmap1,ave1,std1, out1);

		const int Dx2[K]={
	#include "dx2"
				};
				const int Dy2[K]={
	#include "dy2"
				};
				const DataType p2[K][N]={
	#include "p2"
				};

				const DataType ave2[N]={
	#include "ave2"
				};
				const DataType std2[N]={
	#include "std2"
				};

		DataType fmap2[nD][nD][N];
		_shift_s<DataType, D, K, N, S>(out1, fmap2, Dx2, Dy2, p2);
		_norm_relu<DataType, nD, N>(fmap2, ave2,std2, act);
}



