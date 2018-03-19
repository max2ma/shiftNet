#pragma once

#include "hls_half.h"
template<typename T, int Df, int M, int N>
void _shift(T fmap[Df][Df][M], T act[Df][Df][N],
		const int Dx[M],
		const int Dy[M],
		const T p[M][N]){
#pragma HLS INLINE

	_shift_loopI:for(int i=0;i<Df;i++){
		_shift_loopJ:for(int j=0;j<Df;j++){
			_shift_loopN:for(int n=0;n<N;n++){
				half sum = 0;
				_shift_loopM:for(int m=0;m<M;m++){
					int dx = i - Dx[m];
					int dy = j - Dy[m];
					if(dx<0 || dx>=Df || dy<0 || dy>=Df)
						continue;
					sum += p[m][n].to_half() * fmap[dx][dy][m].to_half();
				}
				act[i][j][n] = T(sum);
			}
		}
	}
}

template<typename T, int Df, int M, int N>
void _shift_n(T fmap[Df][Df][M], T act[Df][Df][N],
		const int Dx[M],
		const int Dy[M],
		const T p[M][N]){
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=p complete dim=1
	T sum[N];
#pragma HLS ARRAY_PARTITION variable=sum complete dim=1
	for(int i=0;i<N;i++)
		sum[i] = 0;

	for(int i=0;i<Df;i++){
		for(int j=0;j<Df;j++){
			for(int m=0;m<M;m++){
#pragma HLS PIPELINE
				int dx = i - Dx[m];
				int dy = j - Dy[m];
				if(dx<0 || dx>=Df || dy<0 || dy>=Df)
					continue;
				for(int n=0;n<N;n++){
					sum[n] += p[m][n] * fmap[dx][dy][m];
				}

			}
			for(int n=0;n<N;n++){
#pragma HLS PIPELINE
				act[i][j][n] = sum[n];
				sum[n] = 0;
			}
		}
	}
}


template<typename T, int Df, int M, int N, int S>
void _shift_s(T fmap[Df][Df][M], T act[(Df - 1)/S + 1][(Df - 1)/S + 1][N],
		const int Dx[M],
		const int Dy[M],
		const T p[M][N]){
#pragma HLS ARRAY_PARTITION variable=p complete dim=2
	T sum[N];
#pragma HLS ARRAY_PARTITION variable=sum complete dim=1
	for(int i=0;i<N;i++)
		sum[i] = 0;

	static const int nD = (Df - 1)/S + 1;

	for(int i=0;i<nD;i++){
		for(int j=0;j<nD;j++){
			for(int m=0;m<M;m++){
#pragma HLS PIPELINE
				int dx = i * S - Dx[m];
				int dy = j * S - Dy[m];
				if(dx<0 || dx>=Df || dy<0 || dy>=Df)
					continue;
				for(int n=0;n<N;n++){
					sum[n] += p[m][n] * fmap[dx][dy][m];
				}
			}
			for(int n=0;n<N;n++){
#pragma HLS PIPELINE
				act[i][j][n] = sum[n];
				sum[n] = 0;
			}
		}
	}
}

template<typename T, int Df, int M, int N, int S>
void _shift_r(T fmap[Df][Df][M], T act[(Df - 1)/S + 1][(Df - 1)/S + 1][N],
		const int Dx[M],
		const int Dy[M],
		const T p[M][N]){
#pragma HLS ARRAY_PARTITION variable=p complete dim=1
	T sum[N];
#pragma HLS ARRAY_PARTITION variable=sum complete dim=1
	for(int i=0;i<N;i++)
		sum[i] = 0;

	static const int nD = (Df - 1)/S + 1;

	for(int i=0;i<nD;i++){
		for(int j=0;j<nD;j++){
			for(int m=0;m<M;m++){
#pragma HLS PIPELINE
				int dx = i * S - Dx[m];
				int dy = j * S - Dy[m];
				if(dx<0 || dx>=Df || dy<0 || dy>=Df)
					continue;
				for(int n=0;n<N;n++){
					sum[n] += p[m][n] * fmap[dx][dy][m];
				}

			}
			for(int n=0;n<N;n++){
#pragma HLS PIPELINE
				act[i][j][n] = sum[n] > 0? sum[n] : 0;
				sum[n] = 0;
			}
		}
	}
}
