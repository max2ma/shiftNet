#include <iostream>
#include <cmath>
#include "hls_stream.h"
#include "para.h"
using namespace std;
using namespace para;

extern
void shift(hls::stream<DataType> *tensor, hls::stream<DataType>* act);

int main(){
	DataType input[D][D][C] = {
#include "input"
	};

	float ref[nD][nD][N] = {
#include "t_act"
	};


hls::stream<DataType> istream[C], ostream[N];
for(int i=0;i<D;i++)
		for(int j=0;j<D;j++)
			for(int k=0;k<C;k++)
				istream[k].write(input[i][j][k]);


	shift(istream, ostream);


	int err = 0;
	float ave = 0;
	for(int i=0;i<nD;i++)
		for(int j=0;j<nD;j++)
			for(int k=0;k<N;k++){
				DataType output = ostream[k].read();
#ifdef FIXED
				float diff = abs(( output - (DataType)ref[i][j][k]).to_float());
				ave+=diff;
#else
				float diff = abs(output - ref[i][j][k]);
				ave+=diff;
#endif
				if (diff > 1e-2){
					err ++;
					cout <<i<<','
							<<j<<','
							<<k<<','
							<<output<<','
							<<ref[i][j][k]<<endl;

				}
			}
	cout << "there are in total " << err << " errors."<<endl;
	cout << "the ave error is " << ave/D/D/N << " ."<<endl;
	if(err == 0)
		return 0;
	else
		return -1;
}
