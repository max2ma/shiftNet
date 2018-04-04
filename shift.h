#pragma once

#include "hls_stream.h"
#include "buffer.h"

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
	ShiftBuffer<T,2,D,C> buffer;
	for(int i=0;i<D;i++)
		for(int j=0;j<C;j++){
#pragma HLS PIPELINE
			T r = fmap.read();
			buffer.insert(r, j);
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

				switch(Dy[k]){
					case 0:
						switch(Dx[k]){
							case 0:
								buffer.ih[k] = 0;
								w = buffer.insert(r,k);
								break;
							case 1:
								w = buffer.insert(r,k);
								break;
							case -1:
								w = r;
//								buffer.insert(r,k);
								break;
						}
						break;
					case 1:
						if(j ==0) {
							buffer.ih[k] = 0;
							buffer.pixel[k].shift_pixels_right();
							buffer.pixel[k].insert_pixel(0, 0, 0);
						}
						w = buffer.insert(r,k);
						break;
					case -1:
						if(j ==0) {
							buffer.ih[k] = 0;
							buffer.pixel[k].shift_pixels_left();
							buffer.pixel[k].insert_pixel(0, 0, D - 1);
						}
						w = buffer.insert(r,k);
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
	for(int i = 0, is = 0;i<D;i++, is++)
		for(int j = 0, js = 0, c = 0;j<D;j++, js++)
			for(int k = 0;k<C;k++){
#pragma HLS PIPELINE
				T r = fmap.read();
				if(is == S) is = 0;
				if(js == S) {
					js = 0;
					c++;
				}
				if((is == 0 && js ==0) || buffer[c][k] < r)
					buffer[c][k] = r;
				if( is == S - 1 && js == S -1)
					act.write(buffer[c][k]);
			}
}
