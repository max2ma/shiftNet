#pragma once
#include "hls_stream.h"



template<typename T, int D, int C, int P, int IIs>
void padding(hls::stream<T> fmap[C], hls::stream<T> omap[C]){
#pragma HLS INLINE
	
	for(int i=0;i<D + 2*P;i++)
		for(int j=0;j<D + 2*P;j++){
#pragma HLS PIPELINE
			bool pad = (i < P) || (i >= D + P) || (j < P) || (j >= D + P);
			for(int c=0;c<C;c++){
				if(pad)
					omap[c].write(0);
				else
					omap[c].write(fmap[c].read());
			}
		}
}

template<typename T, int D, int C, int K, int S, int IIs>
void conv2d_3x3(hls::stream<T> fmap[C], const T kernel[3][3][C][K], hls::stream<T> omap[K]){
#pragma HLS ARRAY_PARTITION variable=kernel complete dim=1
#pragma HLS ARRAY_PARTITION variable=kernel complete dim=2
#pragma HLS ARRAY_PARTITION variable=kernel complete dim=3
#pragma HLS ARRAY_PARTITION variable=kernel complete dim=4

#pragma HLS INLINE

	static const int nD = (D - 3)/S + 1;
	T buffer[2][D][C];
#pragma HLS ARRAY_PARTITION variable=buffer complete dim=3
#pragma HLS ARRAY_PARTITION variable=buffer complete dim=1
	T crop[3][3][C];
#pragma HLS ARRAY_PARTITION variable=crop complete dim=0
	T sum[K];
#pragma HLS ARRAY_PARTITION variable=sum complete dim=0
/*
	for(int i=0;i<2;i++)
		for(int j=0;j<D;j++)
#pragma HLS PIPELINE
			for(int c=0;c<C;c++)
				buffer[i][j][c] = fmap[c].read();
*/
	for(int i=0;i<D;i++){
		for(int j=0;j<D;j++){
#pragma HLS PIPELINE II=IIs
			int ci = (i & 0x01);
			bool b_out = ((i-2) % S == 0) && ((j - 2) % S == 0);
			for(int c = 0;c<C;c++){
#pragma HLS PIPELINE
				// crop shift left
				for(int si = 0; si <3;si++)
					for(int sj = 0; sj <2;sj++)
#pragma HLS PIPELINE
						crop[si][sj][c] = crop[si][sj+1][c];
				// crop read buffer
				for(int si = 0; si < 2;si++){
#pragma HLS PIPELINE
					int bi = (ci + si) & 0x01;
					crop[si][2][c] = buffer[bi][j][c];
				}
				// read from fmap and put to crop and buffer
				crop[2][2][c] = fmap[c].read();
				buffer[ci][j][c] = crop[2][2][c];
			}
			for(int k=0;k<K;k++){
#pragma HLS PIPELINE
				sum[k] = 0;
				for(int c = 0;c<C;c++)
					for(int fi = 0; fi<3;fi++)
						for(int fj = 0; fj<3;fj++)
							sum[k] += crop[fi][fj][c] * kernel[fi][fj][c][k];
				if(j>=2 && i>=2 && b_out)
					omap[k].write(sum[k]);
			}
		}
	}

}
template<typename T, int D, int C, int F, int K, int S, int IIs>
void conv2d(hls::stream<T> fmap[C], const T kernel[F][F][C][K], hls::stream<T> omap[K]){

#pragma HLS INLINE

	static const int nD = (D - F)/S + 1;
	T buffer[F-1][D][C];
#pragma HLS ARRAY_PARTITION variable=buffer complete dim=3
#pragma HLS ARRAY_PARTITION variable=buffer complete dim=1
	T crop[F][F][C];
#pragma HLS ARRAY_PARTITION variable=crop complete dim=0
	T sum[K];
//#pragma HLS ARRAY_PARTITION variable=sum complete dim=0
/*
	for(int i=0;i<F-1;i++)
		for(int j=0;j<D;j++)
			for(int c=0;c<C;c++)
#pragma HLS PIPELINE
				buffer[i][j][c] = fmap[c].read();
*/
	for(int i=0, ci = 0;i<D;ci++, i++){
		for(int j=0;j<D;j++){
#pragma HLS PIPELINE II=IIs
			if(ci == F - 1) ci = 0;
			bool b_out = (i >=F-1) && ((i-F+1) % S == 0) && (j >= F - 1) && ((j - F + 1) % S == 0);
			for(int c = 0;c<C;c++){
#pragma HLS PIPELINE
				// crop shift left
				for(int si = 0; si <F;si++)
					for(int sj = 0; sj <F - 1;sj++)
#pragma HLS PIPELINE
						crop[si][sj][c] = crop[si][sj+1][c];
				// crop read buffer
				for(int si = 0, sci=ci; si < F - 1;si++, sci ++){
#pragma HLS PIPELINE
					if(sci == F - 1)sci = 0;
					crop[si][F -1][c] = buffer[sci][j][c];
				}
				// read from fmap and put to crop and buffer
				crop[F-1][F -1][c] = fmap[c].read();
				buffer[ci][j][c] = crop[F-1][F-1][c];
			}
			for(int k=0;k<K;k++){
#pragma HLS PIPELINE
				sum[k] = 0;
				for(int c = 0;c<C;c++)
					for(int fi = 0; fi<F;fi++)
						for(int fj = 0; fj<F;fj++)
							sum[k] += crop[fi][fj][c] * kernel[fi][fj][c][k];
				if(b_out)
					omap[k].write(sum[k]);
			}
		}
	}

}
