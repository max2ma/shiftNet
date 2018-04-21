#pragma once

#include "hls_stream.h"

namespace SingleChan{

	template<typename T, int D, int C, int N>
	void _matMul(hls::stream<T> &fmap, hls::stream<T> &out, const T p[D*D*C][N]){
#pragma HLS ARRAY_PARTITION variable=p complete dim=2
		T sum[N];
#pragma HLS ARRAY_PARTITION variable=sum complete dim=1
		for(int n=0;n<N;n++)
#pragma HLS UNROLL
			sum[n] = 0;
		for(int i=0;i<D;i++){
			for(int j=0;j<D;j++){
				for(int k=0;k<C;k++){
#pragma HLS PIPELINE
					T r = fmap.read();
					for(int n=0;n<N;n++)
						sum[n] += p[i*D*C + j*C+ k][n] * r;
				}
			}
		}
		for(int n=0;n<N;n++)
#pragma HLS PIPELINE
			out.write(sum[n]);
	}

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
				const int Dx[C]){
			static const int nD = (D - 1)/S + 1;
			T buffer[2][D][C];
			for(int i=0;i<D;i++)
				for(int j=0;j<C;j++){
#pragma HLS PIPELINE
					T r = fmap.read();
					buffer[0][i][j] = r;
					buffer[1][i][j] = 0;
				}
			for(int i=0;i<D;i++)
				for(int j=0;j<D;j++)
					for(int k=0;k<C;k++){
#pragma HLS PIPELINE
						T r,w;
						if(i != D -1)
							r = fmap.read();
						else
							r = 0;

						int ci = i & 0x01;
						int ni = ci ^ 0x01;

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
							case 2:
								buffer[ni][j][k] = r;
								if(j == 0) {
									w = 0;
								} else
									w = buffer[ci][j - 1][k];
								break;
							case -2:
								buffer[ni][j][k] = r;
								if(j == D - 1) {
									w = 0;
								} else
									w = buffer[ci][j + 1][k];
								break;
							default:
								break;
						}
						if(i % S == 0 && j % S ==0)
							act.write(w);
					}
		}

	template<typename T, int D, int C, int N, int S>
		void _conv2d_1x1(hls::stream<T> &fmap, hls::stream<T> &out, const T p[C][N]){
#pragma HLS ARRAY_PARTITION variable=p complete dim=2

			T sum[N];
#pragma HLS ARRAY_PARTITION variable=sum complete dim=1
			T tmp[N];
#pragma HLS ARRAY_PARTITION variable=tmp complete dim=1
			for(int i=0;i<N;i++)
#pragma HLS PIPELINE
				sum[i] = 0;

			int is =0, js =0;
			for(int i=0, is =0;i<D;i++, is++){
				for(int j=0, js=0;j<D;j++, js++){

#pragma HLS PIPELINE II=C
					if(is == S) is =0;
					if(js == S) js =0;

					for(int k=0;k<C;k++){
						tmp[k] = fmap.read();
						for(int n=0;n<N;n++)
							sum[n] += p[k][n] * tmp[k];
					}
					for(int n=0;n<N;n++){
						if(is ==0 && js ==0)
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

	template<typename T, int D, int C1, int C2>
		void _concat(hls::stream<T> & in1, hls::stream<T>& in2, hls::stream<T> &out){
			for(int i=0;i<D;i++)
				for(int j=0;j<D;j++){
					for(int k = 0; k<C1;k++)
#pragma HLS PIPELINE
						out.write(in1.read());
					for(int k = 0; k<C2;k++)
#pragma HLS PIPELINE
						out.write(in2.read());
				}
		}

	template<typename T, int D, int C>
		void _duplicate(hls::stream<T> & in, hls::stream<T>& out1, hls::stream<T> &out2){
			for(int i=0;i<D;i++)
				for(int j=0;j<D;j++)
					for(int k = 0; k<C;k++){
#pragma HLS PIPELINE
						T r = in.read();
						out1.write(r);
						out2.write(r);
					}
		}

	template<typename T, int D, int D_shift, int S_conv, int IP, int E, int OP>
		void _shift(hls::stream<T> & input,
				hls::stream<T> & output,
				const int Dx[IP * E],
				const T p0[IP][IP * E],
				const T p1[IP * E][OP]
				){
			static const int MP = IP * E;
			static const int sD = (D - 1)/D_shift + 1;
			static const int cD = (sD - 1)/S_conv + 1;

#pragma HLS INLINE
			hls::stream<DataType> f_conv0, f_shift,f_conv1, f_relu;
			_conv2d_1x1<DataType,D, IP, MP, 1>(input, f_conv0, p0);
			_relu<DataType, D, MP>(f_conv0, f_relu);
			_shift_3x3<DataType, D, MP, D_shift>(f_relu, f_shift, Dx);
			_conv2d_1x1<DataType,sD, MP, OP, S_conv>(f_shift, f_conv1, p1);
			_relu<DataType, cD, OP>(f_conv1, output);
		}

	template<typename T, int D, int D_shift, int S_conv, int IP, int MP, int OP>
		void _shift_pool(hls::stream<T> & input,
				hls::stream<T> & output,
				const int Dx[MP],
				const T p0[IP][MP],
				const T p1[MP][OP]
				){
			static const int sD = (D - 1)/D_shift + 1;
			static const int cD = (sD - 1)/S_conv + 1;

#pragma HLS INLINE
			hls::stream<DataType> f_in1, f_in2, f_conv0, f_shift,f_conv1, f_pool, f_relu;
#pragma HLS STREAM variable=f_in1 depth=1 dim=1
#pragma HLS STREAM variable=f_in2 depth=1 dim=1
#pragma HLS STREAM variable=f_shift depth=1 dim=1
#pragma HLS STREAM variable=f_conv0 depth=1 dim=1
#pragma HLS STREAM variable=f_conv1 depth=1 dim=1
#pragma HLS STREAM variable=f_pool depth=1 dim=1
#pragma HLS STREAM variable=f_relu depth=1 dim=1

			_duplicate<T, D, IP>(input, f_in1, f_in2);
			_conv2d_1x1<DataType,D, IP, MP, 1>(f_in1, f_conv0, p0);
			_relu<DataType, D, OP>(f_conv0, f_relu);
			_shift_3x3<DataType, D, MP, D_shift>(f_relu, f_shift, Dx);
			_conv2d_1x1<DataType,sD, MP, OP, S_conv>(f_shift, f_conv1, p1);
			_max_pool<DataType, D, OP, D_shift * S_conv>(f_in2,f_pool);
			_concat<T, cD, OP, IP>(f_conv1, f_pool, output);

		}

}





