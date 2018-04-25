#pragma once

#include "hls_stream.h"

namespace MulChan{

	template<typename T, int D, int C, int S, int IIs>
		void _shift_3x3(hls::stream<T> fmap[C], hls::stream<T> out[C], const int Dx[C]){
			static const int nD = (D - 1)/S + 1;
#pragma HLS ARRAY_PARTITION variable=Dx complete dim=1
#pragma HLS ARRAY_PARTITION variable=fmap complete dim=1
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
			T buffer[2][D][C];
#pragma HLS ARRAY_PARTITION variable=buffer complete dim=3
			for(int i=0;i<D;i++)
#pragma HLS PIPELINE
				for(int j=0;j<C;j++){
					T r = fmap[j].read();
					buffer[0][i][j] = r;
					buffer[1][i][j] = 0;
				}
			for(int i=0;i<D;i++){
				for(int j=0;j<D;j++){
#pragma HLS PIPELINE II = IIs
					for(int k=0;k<C;k++){
						T r,w;
						if(i != D -1)
							r = fmap[k].read();
						else
							r = 0;

						int ci = i & 0x01;
						int ni = ci ^ 0x01;

						switch(Dx[k]){
							case 0:// No MOVE
								w = buffer[ci][j][k];
								buffer[ni][j][k] = r;
								break;
							case 1:// MOVE DOWN
								w = buffer[ni][j][k];
								buffer[ni][j][k] = r;
								break;
							case -1: // MOVE UP
								w = r;
								break;
							case 2: // MOVE RIGHT
								buffer[ni][j][k] = r;
								if(j == 0) {
									w = 0;
								} else
									w = buffer[ci][j - 1][k];
								break;
							case -2:// MOVE LEFT
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
							out[k].write(w);
					}
				}
			}
		}
	template<typename T, int D, int C, int S, int IIs>
		void _max_pool(hls::stream<T> fmap[C], hls::stream<T> out[C]){
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
#pragma HLS ARRAY_PARTITION variable=fmap complete dim=1
			static const int nD = D/S;
			T buffer[nD][C];
#pragma HLS ARRAY_PARTITION variable=buffer complete dim=2
			int c =0, is =0, js =0;
			for(int i = 0;i<D;i++, is++)
				for(int j = 0, js = 0, c = 0;j<D;j++, js++)
#pragma HLS PIPELINE II=IIs
					for(int k = 0;k<C;k++){
						if(is == S) is = 0;
						if(js == S) {
							js = 0;
							c++;
						}
						if(c == nD){
							fmap[k].read();
							continue;
						}
						T r = fmap[k].read();
						T cmp = buffer[c][k];
						if((is == 0 && js ==0) || cmp < r){
							buffer[c][k] = r;
							cmp = r;
						}
						if( is == S - 1 && js == S -1)
							out[k].write(cmp);
					}
		}
	
	template<typename T, int D, int C, int IIs>
		void _bias_add(hls::stream<T> fmap[C], const T bias[C], hls::stream<T> out[C]){
#pragma HLS ARRAY_PARTITION variable=fmap complete dim=1
#pragma HLS ARRAY_PARTITION variable=bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
			for(int i=0;i<D;i++){
				for(int j=0;j<D;j++){
#pragma HLS PIPELINE II=IIs
					for(int k=0;k<C;k++){
						out[k].write(fmap[k].read() + bias[k]);
					}
				}
			}
		}
	template<typename T, int D, int C, int IIs>
		void _add(hls::stream<T> fmap_0[C], hls::stream<T> fmap_1[C], hls::stream<T> out[C]){
#pragma HLS ARRAY_PARTITION variable=fmap_0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=fmap_1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
			for(int i=0;i<D;i++){
				for(int j=0;j<D;j++){
#pragma HLS PIPELINE II=IIs
					for(int k=0;k<C;k++){
						out[k].write(fmap_0[k].read() + fmap_1[k].read());
					}
				}
			}
		}
	
	template<typename T, int D, int C, int IIs>
		void _duplicate(hls::stream<T> fmap[C], hls::stream<T> out_0[C], hls::stream<T> out_1[C]){
#pragma HLS ARRAY_PARTITION variable=fmap complete dim=1
#pragma HLS ARRAY_PARTITION variable=out_0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=out_1 complete dim=1
			for(int i=0;i<D;i++){
				for(int j=0;j<D;j++){
#pragma HLS PIPELINE II=IIs
					for(int k=0;k<C;k++){
						T r = fmap[k].read();
						out_0[k].write(r);
						out_1[k].write(r);
					}
				}
			}
		}
	template<typename T, int D, int C, int IIs>
		void _relu(hls::stream<T> fmap[C], hls::stream<T> out[C]){
#pragma HLS ARRAY_PARTITION variable=fmap complete dim=1
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
			for(int i=0;i<D;i++){
				for(int j=0;j<D;j++){
#pragma HLS PIPELINE II=IIs
					for(int k=0;k<C;k++){
						T diff = fmap[k].read();
						if(diff < 0)
							out[k].write(0);
						else
							out[k].write(diff);
					}
				}
			}
		}

	template<typename T, int D, int C, int N, int S, int IIs>
		void _conv2d_1x1(hls::stream<T> fmap[C], hls::stream<T> out[N], const T p[C][N]){

#pragma HLS ARRAY_PARTITION variable=p complete dim=0
#pragma HLS ARRAY_PARTITION variable=fmap complete dim=1
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
			T sum[N], tmp[C];
#pragma HLS ARRAY_PARTITION variable=sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=tmp complete dim=1

			for(int i=0;i<N;i++)
#pragma HLS PIPELINE
				sum[i] = 0;

			int is =0, js =0;
			for(int i=0, is =0;i<D;i++, is++){
				for(int j=0, js=0;j<D;j++, js++){
#pragma HLS PIPELINE II=IIs
					if(is == S) is =0;
					if(js == S) js =0;

					for(int c=0;c<C;c++)
						tmp[c] = fmap[c].read();
					for(int c=0;c<C;c++)
						for(int n=0;n<N;n++)
							sum[n] += p[c][n] * tmp[c];

					for(int n=0;n<N;n++){
						if(is ==0 && js ==0)
							out[n].write(sum[n]);
						sum[n] = 0;
					}
				}
			}
		}
	template<typename T, int D, int C, int N, int IIs>
	void _matMul(hls::stream<T> fmap[C], hls::stream<T> out[N], const T p[D*D*C][N]){
#pragma HLS ARRAY_PARTITION variable=p complete dim=2
		T sum[N], r[C];
#pragma HLS ARRAY_PARTITION variable=sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=r complete dim=1
		for(int n=0;n<N;n++)
#pragma HLS UNROLL
			sum[n] = 0;
		for(int i=0;i<D;i++){
			for(int j=0;j<D;j++){
#pragma HLS PIPELINE II=IIs
				for(int k=0;k<C;k++){
					r[k] = fmap[k].read();
					for(int n=0;n<N;n++)
						sum[n] += p[i*D*C + j*C+ k][n] * r[k];
				}
			}
		}
		for(int n=0;n<N;n++)
#pragma HLS PIPELINE
			out[n].write(sum[n]);
	}

	template<typename T, int D, int IP, int E, int IIs>
		void _shift(hls::stream<T> input[IP],
				hls::stream<T> output[IP],
				const int Dx[IP * E],
				const T p0[IP][IP * E],
				const T p1[IP * E][IP],
				const T bias0[IP*E],
				const T bias1[IP]
				){
			static const int MP = IP * E;

#pragma HLS INLINE
			hls::stream<DataType> f_conv0[MP], f_bias0[MP], f_shift[MP],f_conv1[IP], f_bias1[IP], f_relu[MP];

			_conv2d_1x1<DataType,D, IP, MP, 1, IIs>(input, f_conv0, p0);
			_bias_add<DataType, D, MP, IIs>(f_conv0, bias0, f_bias0);
			_relu<DataType, D, MP, IIs>(f_bias0, f_relu);
			_shift_3x3<DataType, D, MP, 1, IIs>(f_relu, f_shift, Dx);
			_conv2d_1x1<DataType,D, MP, IP, 1, IIs>(f_shift, f_conv1, p1);
			_bias_add<DataType, D, IP, IIs>(f_conv1, bias1,f_bias1);
			_relu<DataType, D, IP, IIs>(f_bias1, output);
		}
	template<typename T, int D, int S_conv, int IP, int E, int OP, int IIs>
		void _shift_res(hls::stream<T> input[IP],
				hls::stream<T> output[OP],
				const int Dx[IP * E],
				const T p0[IP][IP * E],
				const T p1[IP * E][OP],
				const T p2[IP][OP],
				const T bias0[IP*E],
				const T bias1[OP]
				){
			static const int MP = IP * E;
			static const int nD = (D - 1)/S_conv + 1;

#pragma HLS INLINE
			hls::stream<DataType> f_in0[IP], f_in1[IP];
			_duplicate<DataType, D, IP, IIs>(input, f_in0, f_in1);

			hls::stream<DataType> f_conv0[MP],f_bias0[MP], f_shift[MP], f_relu[MP];
			hls::stream<DataType> f_conv1[OP],f_bias1[OP], f_relu1[OP], f_shortcut[OP];

			_conv2d_1x1<DataType,D, IP, MP, 1, IIs>(f_in0, f_conv0, p0);
			_bias_add<DataType, D, MP, IIs>(f_conv0, bias0, f_bias0);
			_relu<DataType, D, MP, IIs>(f_bias0, f_relu);
			_shift_3x3<DataType, D, MP, 1, IIs>(f_relu, f_shift, Dx);
			_conv2d_1x1<DataType,D, MP, OP, S_conv, IIs>(f_shift, f_conv1, p1);
			_bias_add<DataType, nD, OP, IIs>(f_conv1, bias1,f_bias1);
			_relu<DataType, nD, OP, IIs>(f_bias1, f_relu1);

			_conv2d_1x1<DataType, D, IP, OP, S_conv, IIs>(f_in1, f_shortcut, p2);
			_add<DataType, nD, OP, IIs>(f_relu1, f_shortcut, output);
		}
}

