#pragma once

#include "hls_stream.h"

template<typename T, int D, int C>
void _relu(hls::stream<T>& fmap, hls::stream<T>& act){
	for(int i=0;i<D;i++){
		for(int j=0;j<D;j++){
			for(int k=0;k<C;k++){
#pragma HLS PIPELINE
				T diff = fmap.read();
				if(diff < 0)
					act.write(0);
				else
					act.write(diff);
			}
		}
	}
}

template<typename T, int D, int C, int S>
void _shift_3x3(hls::stream<T>& fmap, hls::stream<T>& act,
		const int Dx[C],
		const int Dy[C]){
	static const int nD = (D - 1)/S + 1;
	T buffer[2][D][C];
#pragma HLS ARRAY_PARTITION variable=buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=buffer complete dim=2
	for(int i=0;i<D;i++)
		for(int j=0;j<C;j++){
#pragma HLS PIPELINE
			T r = fmap.read();
			buffer[0][i][j] = r;
			buffer[1][i][j] = 0;
		}
	for(int i=0;i<D;i++){
		for(int j=0;j<D;j++){
			for(int k=0;k<C;k++){
#pragma HLS PIPELINE
				T r,w;
				if(i != D -1)
					r = fmap.read();
				else
					r = 0;

				int ci = i & 0x01;
				int ni = ci ^ 0x01;

				switch(Dy[k]){
					case 0:
						switch(Dx[k]){
							case 0:
								w = buffer[ci][j][k];
								buffer[ni][j][k] = r;
								break;
							case 1:
								w = buffer[ni][j][k];
								buffer[ni][j][k] = r;
								break;
							case -1:
								w = r;
								break;
						}
						break;
					case 1:
						buffer[ni][j][k] = r;
						if(j == 0) {
							w = 0;
						} else
							w = buffer[ci][j - 1][k];
						break;
					case -1:
						buffer[ni][j][k] = r;
						if(j == D - 1) {
							w = 0;
						} else
							w = buffer[ci][j + 1][k];
						break;
				}
				if(i % S == 0 && j % S ==0)
					act.write(w);
			}
		}
	}
}

template<typename T,
int D, int C, int N>
void _linear_combination(hls::stream<T> &fmap, hls::stream<T> &out, const T p[C][N]){

	T sum[N];
#pragma HLS ARRAY_PARTITION variable=sum complete dim=1

	for(int i=0;i<N;i++)
#pragma HLS PIPELINE
		sum[i] = 0;

	for(int i=0;i<D;i++){
		for(int j=0;j<D;j++){
			for(int k=0;k<C;k++){
#pragma HLS PIPELINE
				T tmp = fmap.read();
				for(int n=0;n<N;n++)
#pragma HLS UNROLL
					sum[n] += p[k][n] * tmp;
			}
			for(int n=0;n<N;n++){
#pragma HLS PIPELINE
				out.write(sum[n]);
				sum[n] = 0;
			}
		}
	}
}


template<typename T, int D, int C, int S>
void _max_pool(hls::stream<T> &fmap, hls::stream<T> &act){

	static const int nD = D/S;
	T buffer[nD][C];
	int c =0, is =0, js =0;
	for(int i = 0;i<D;i++, is++)
		for(int j = 0, js = 0, c = 0;j<D;j++, js++)
			for(int k = 0;k<C;k++){
#pragma HLS PIPELINE
				if(is == S) is = 0;
				if(js == S) {
					js = 0;
					c++;
				}
				if(c == nD){
					fmap.read();
					continue;
				}
				T r = fmap.read();
				T cmp = buffer[c][k];
				if((is == 0 && js ==0) || cmp < r){
					buffer[c][k] = r;
					cmp = r;
				}
				if( is == S - 1 && js == S -1)
					act.write(cmp);
			}
}
