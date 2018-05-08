#include <iostream>
#include <fstream>
#include <cmath>
#include "hls_stream.h"
#include "para.h"
using namespace std;
using namespace para;

extern
void shift(hls::stream<DataType> *tensor, hls::stream<DataType>* act);

int main(){
	DataType input[D][D][C] = {
#include "inputs_batch_0"
		//#include "t_im"
	};

	float ref[N] = {
#include "outputs_batch_0"
		//#include "t_cifar"
	};

	hls::stream<DataType> istream[C], ostream[N];
	int err = 0, TT =N;
	float ave = 0;

	DataType in, r;
	char c;

	for(int i=0;i<D;i++){
		for(int j=0;j<D;j++){
			for(int k=0;k<C;k++){
				istream[k].write(input[i][j][k]);
			}
		}
	}
	shift(istream, ostream);



	for(int k=0;k<N;k++){
		r = ref[k];
		DataType output = ostream[k].read();
#ifdef FIXED
		float diff = abs(( output - (DataType)r).to_float());
		ave+=diff;
#else
		if(r == 0.0){
			TT --;
			if(output == 0.0)
				continue;
			else
				err ++;
		}
		float diff = abs(output / r - 1);// ref[i][j][k]);
		ave+=diff;
#endif
		if (diff > 1e-1)
			err ++;
		cout	<<k<<','
			<<output<<','
			<<r << ','
			<<endl;
	}
	cout << "there are in total " << err << " errors."<<endl;
	cout << "the ave error is " << ave/TT << " ."<<endl;
	if(err == 0){
		return 0;
	}
	else {
		return -1;
	}
}
