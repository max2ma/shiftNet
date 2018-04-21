#pragma once
#include "hls_stream.h"



template<typename T, int D, int C, int P>
void padding(hls::stream<T> &fmap, hls::stream<T> &omap){
#pragma HLS INLINE
	
	for(int i=0;i<D + 2*P;i++)
		for(int j=0;j<D + 2*P;j++){
			bool pad = (i < P) || (i >= D + P) || (j < P) || (j >= D + P);
			for(int c=0;c<C;c++){
#pragma HLS PIPELINE
				if(pad)
					omap.write(0);
				else
					omap.write(fmap.read());
			}
		}
}

template<typename T, int D, int C, int F, int K, int S>
void conv2d(hls::stream<T> &fmap, const T kernel[F][F][C][K], hls::stream<T> &omap){

#pragma HLS INLINE

	static const int nD = (D - F)/S + 1;
	T buffer[F-1][D][C];
#pragma HLS ARRAY_PARTITION variable=buffer complete dim=3
#pragma HLS ARRAY_PARTITION variable=buffer complete dim=1
	T crop[F][F][C];
#pragma HLS ARRAY_PARTITION variable=crop complete dim=0
	T sum[K];
	for(int i=0, ci = 0;i<D;ci++, i++){
		for(int j=0;j<D;j++){
#pragma HLS PIPELINE 
			if(ci == F - 1) ci = 0;
			bool b_out = (i >=F-1) && ((i-F+1) % S == 0) && (j >= F - 1) && ((j - F + 1) % S == 0);
			for(int c = 0;c<C;c++){
#pragma HLS PIPELINE
				// crop shift left
				for(int si = 0; si <F;si++)
					for(int sj = 0; sj <F - 1;sj++)
						crop[si][sj][c] = crop[si][sj+1][c];
				// crop read buffer
				for(int si = 0, sci=ci; si < F - 1;si++, sci ++){
					if(sci == F - 1)sci = 0;
					crop[si][F -1][c] = buffer[sci][j][c];
				}
				// read from fmap and put to crop and buffer
				crop[F-1][F -1][c] = fmap.read();
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
					omap.write(sum[k]);
			}
		}
	}

}
