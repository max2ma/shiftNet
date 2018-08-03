#pragma once

#include "utils/x_hls_traits.h"
#include "hls_stream.h"
#include "log2.h"

namespace MulChan{

	template<int D, int C, int S, int IIs, int REP, typename T>
		void _shift_3x3(hls::stream<T> fmap[C], hls::stream<T> omap[C]){
#pragma HLS INLINE

			const static int G = (C/9) * 9;
			T buffer[2][D+2][C];
#pragma HLS ARRAY_PARTITION variable=buffer complete dim=3
#pragma HLS ARRAY_PARTITION variable=buffer complete dim=1
			T crop[3][3][C];
#pragma HLS ARRAY_PARTITION variable=crop complete dim=0
			T value;
			for(int rep = 0; rep < REP; rep++){
				for(int i=0;i<D + 2;i++){
					for(int j=0;j<D + 2;j++){
#pragma HLS PIPELINE II=IIs
						int ci = (i & 0x01);
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

							if(i==0 || i == D + 1 || j==0 || j == D+1){
								buffer[ci][j][c] = 0;
								crop[2][2][c] = 0;
							}
							else {
								T r = fmap[c].read();
								crop[2][2][c] = r;
								buffer[ci][j][c] = r;
							}
							if(c>=G)
								value = crop[1][1][c];

							else{
								int d_x = c%3;
								int d_y = (c/3)%3;
								value = crop[d_y][d_x][c];
							}
							if(j>=2 && i>=2)
								omap[c].write(value);
						}
					}
				}
			}
		}
	template<int D, int C, int S, int IIs, int REP, typename T>
		void _max_pool(hls::stream<T> fmap[C], hls::stream<T> out[C]){
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
#pragma HLS ARRAY_PARTITION variable=fmap complete dim=1
			static const int nD = D/S;
			T buffer[nD][C];
#pragma HLS ARRAY_PARTITION variable=buffer complete dim=2
			for(int rep = 0; rep < REP; rep++){
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
		}
	template<int D, int C, int S, int IIs, int REP, typename T>
		void _avg_pool(hls::stream<T> fmap[C], hls::stream<T> out[C]){
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
#pragma HLS ARRAY_PARTITION variable=fmap complete dim=1
			static const int nD = D/S;
			typedef typename hls::x_traits<T, ap_uint<CE_LOG2<S*S>::V> >::MULT_T SUM_T;
			SUM_T buffer[nD][C];
#pragma HLS ARRAY_PARTITION variable=buffer complete dim=2
			for(int rep = 0; rep < REP; rep++){
				int col =0, is =0, js =0;
				for(int i = 0;i<D;i++, is++)
					for(int j = 0, js = 0, col = 0;j<D;j++, js++)
#pragma HLS PIPELINE II=IIs
						for(int c = 0;c<C;c++){
							if(is == S) is = 0;
							if(js == S) {
								js = 0;
								col++;
							}
							if(col == nD){
								fmap[c].read();
								continue;
							}
							T r = fmap[c].read();
							if(is == 0 && js ==0) {
								buffer[col][c] = 0;
							}
							buffer[col][c] +=r;
							if( is == S - 1 && js == S -1)
								out[c].write(buffer[col][c] /(S * S));
						}
			}
		}


	template< int D, int C, int IIs, int REP, typename T1, typename T2, typename T3>
		void _bias_add(hls::stream<T1> fmap[C], const T2 bias[C], hls::stream<T3> out[C]){
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=fmap complete dim=1
#pragma HLS ARRAY_PARTITION variable=bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
			for(int rep = 0; rep < REP; rep++)
				for(int i=0;i<D;i++){
					for(int j=0;j<D;j++){
#pragma HLS PIPELINE II=IIs
						for(int k=0;k<C;k++){
							out[k].write(fmap[k].read() + bias[k]);
						}
					}
				}
		}
	template< int D, int C, int IIs, int REP, typename T1, typename T2, typename T3>
		void _add(hls::stream<T1> fmap_0[C], hls::stream<T2> fmap_1[C], hls::stream<T3> out[C]){
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=fmap_0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=fmap_1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
			for(int rep = 0; rep < REP; rep++){
				for(int i=0;i<D;i++){
					for(int j=0;j<D;j++){
#pragma HLS PIPELINE II=IIs
						for(int k=0;k<C;k++){
							out[k].write(fmap_0[k].read() + fmap_1[k].read());
						}
					}
				}
			}
		}

	template< int D, int C, int REP, typename T>
		void _duplicate(hls::stream<T> fmap[C], hls::stream<T> out_0[C], hls::stream<T> out_1[C]){
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=fmap complete dim=1
#pragma HLS ARRAY_PARTITION variable=out_0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=out_1 complete dim=1
			for(int rep = 0; rep < REP; rep++)
				for(int i=0;i<D;i++){
					for(int j=0;j<D;j++){
#pragma HLS PIPELINE 
						for(int k=0;k<C;k++){
							T r = fmap[k].read();
							out_0[k].write(r);
							out_1[k].write(r);
						}
					}
				}
		}
	template< int D, int C, int IIs, int REP, typename T>
		void _relu(hls::stream<T> fmap[C], hls::stream<T> out[C]){
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=fmap complete dim=1
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
			for(int rep = 0; rep < REP; rep++){
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
		}

	template< int D, int C, int N, int S, int IIs, int REP, typename T_IN, typename T_W, typename T_OUT>
		void _conv2d_1x1(hls::stream<T_IN> fmap[C], hls::stream<T_OUT> out[N], const T_W p[C][N]){
#pragma HLS INLINE

#pragma HLS ARRAY_PARTITION variable=p complete dim=0
#pragma HLS ARRAY_PARTITION variable=fmap complete dim=1
#pragma HLS ARRAY_PARTITION variable=out complete dim=1
			typedef typename hls::x_traits<T_IN, T_W>::MULT_T MULT_T;
			typedef typename hls::x_traits<MULT_T, ap_uint<CE_LOG2<C>::V> >::MULT_T SUM_T;
			SUM_T sum[N];
			T_IN tmp[C];
#pragma HLS ARRAY_PARTITION variable=sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=tmp complete dim=1

			for(int i=0;i<N;i++)
#pragma HLS PIPELINE
				sum[i] = 0;

			for(int rep = 0; rep < REP; rep++)
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
	template< int M, int C, int N, int IIs, int REP, typename T_IN, typename T_W, typename T_OUT>
		void _matMul(hls::stream<T_IN> fmap[C], hls::stream<T_OUT> out[N], const T_W p[M*C][N]){
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=p complete dim=2
			typedef typename hls::x_traits<T_IN, T_W>::MULT_T MULT_T;
			typedef typename hls::x_traits<MULT_T, ap_uint<CE_LOG2<C>::V> >::MULT_T SUM_T;
			SUM_T sum[N], r[C];
#pragma HLS ARRAY_PARTITION variable=sum complete dim=1
#pragma HLS ARRAY_PARTITION variable=r complete dim=1
			for(int n=0;n<N;n++)
#pragma HLS UNROLL
				sum[n] = 0;
			for(int rep = 0; rep < REP; rep++){
				for(int i=0;i<M;i++){
#pragma HLS PIPELINE II=IIs
						for(int k=0;k<C;k++){
							r[k] = fmap[k].read();
							for(int n=0;n<N;n++)
								sum[n] += p[i*C + k][n] * r[k];
						}
				}
				for(int n=0;n<N;n++){
#pragma HLS PIPELINE
					out[n].write(sum[n]);
					sum[n] = 0;
				}
			}
		}

	template<int D, int IP, int E, int IIs, int REP, typename T_S,
		typename T_IN, typename T_OUT, typename T0, typename T1, typename TB0, typename TB1>
		void _shift(hls::stream<T_IN> input[IP],
				hls::stream<T_OUT> output[IP],
				const T0 p0[IP][IP * E],
				const T1 p1[IP * E][IP],
				const TB0 bias0[IP*E],
				const TB1 bias1[IP]
				){
			static const int MP = IP * E;

#pragma HLS INLINE
			hls::stream<T_IN> f_in0[IP], f_in1[IP];
#pragma HLS STREAM VARIABLE=f_in1 DEPTH=D*2
			_duplicate<D, IP, REP>(input, f_in0, f_in1);

			hls::stream<T_S> f_conv0[MP], f_bias0[MP], f_relu[MP],f_shift[MP];
			hls::stream<T_OUT> f_conv1[IP], f_bias1[IP],  f_relu1[IP];

			_conv2d_1x1<D, IP, MP, 1, IIs, REP>(f_in0, f_conv0, p0);
			_bias_add< D, MP, IIs, REP>(f_conv0, bias0, f_bias0);
			_relu< D, MP, IIs, REP>(f_bias0, f_relu);
			_shift_3x3< D, MP, 1, IIs, REP>(f_relu, f_shift);
			_conv2d_1x1<D, MP, IP, 1, IIs, REP>(f_shift, f_conv1, p1);
			_bias_add< D, IP, IIs, REP>(f_conv1, bias1,f_bias1);
			_relu< D, IP, IIs, REP>(f_bias1, f_relu1);

			_add< D, IP, IIs, REP>(f_relu1, f_in1, output);
		}
	template<int D, int S_conv, int IP, int E, int OP, int IIs, int REP, typename T_S,
		typename T_IN, typename T_OUT, typename T0, typename T1, typename T2, typename TB0, typename TB1, typename TB2>
		void _shift_res(hls::stream<T_IN> input[IP],
				hls::stream<T_OUT> output[OP],
				const T0 p0[IP][OP * E],
				const T1 p1[OP * E][OP],
				const T2 p2[IP][OP],
				const TB0 bias0[OP*E],
				const TB1 bias1[OP],
				const TB2 bias2[OP]
				){
			static const int MP = OP * E;
			static const int nD = (D - 1)/S_conv + 1;

#pragma HLS INLINE
			hls::stream<T_IN> f_in0[IP], f_in1[IP];
#pragma HLS STREAM VARIABLE=f_in1 DEPTH=D*2
			_duplicate< D, IP, REP>(input, f_in0, f_in1);

			hls::stream<T_S> f_conv0[MP],f_bias0[MP], f_shift[MP], f_relu[MP];
			hls::stream<T_OUT> f_conv1[OP],f_bias1[OP], f_relu1[OP], f_shortcut[OP], f_bias2[OP];

			_conv2d_1x1<D, IP, MP, 1, IIs, REP>(f_in0, f_conv0, p0);
			_bias_add< D, MP, IIs, REP>(f_conv0, bias0, f_bias0);
			_relu< D, MP, IIs, REP>(f_bias0, f_relu);
			_shift_3x3< D, MP, 1, IIs, REP>(f_relu, f_shift);
			_conv2d_1x1<D, MP, OP, S_conv, IIs, REP>(f_shift, f_conv1, p1);
			_bias_add< nD, OP, IIs, REP>(f_conv1, bias1,f_bias1);
			_relu< nD, OP, IIs, REP>(f_bias1, f_relu1);

			_conv2d_1x1< D, IP, OP, S_conv, IIs, REP>(f_in1, f_shortcut, p2);
			_bias_add< nD, OP, IIs, REP>(f_shortcut, bias2,f_bias2);
			_add< nD, OP, IIs, REP>(f_relu1, f_bias2, output);
		}
}
