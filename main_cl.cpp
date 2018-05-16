/* 
======================================================
 Copyright 2016 Liang Ma

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
======================================================
*
* Author:   Liang Ma (liang-ma@polito.it)
*
*----------------------------------------------------------------------------
*/

#define CL_HPP_ENABLE_EXCEPTIONS

// This should be used when cl.hpp from SDAccel works.
//#include <CL/cl.hpp>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <unistd.h>
#include "xcl2.hpp"
#include "para.h"
using namespace std;
using namespace para;

namespace Params
{
	char *kernel_name=NULL;     // -n
	char *binary_name=NULL;     // -a
}
void usage(char* name)
{
    cout<<"Usage: "<<name
        <<" -b opencl_binary_name"
        <<" -n kernel_name"
        <<endl;
}
int main(int argc, char** argv)
{
	int opt;
	bool flaga=false,flagn=false;
	while((opt=getopt(argc,argv,"n:b:"))!=-1){
		switch(opt){
			case 'n':
				Params::kernel_name=optarg;
				flagn=true;
				break;
			case 'b':
				Params::binary_name=optarg;
				flaga=true;
				break;
			default:
				usage(argv[0]);
				return -1;
		}
	}
	// Check the mandatory argument.
	if(!flagn || !flaga) {
		usage(argv[0]);
		return -1;
	}
	ifstream ifstr(Params::binary_name);
	const string programString(istreambuf_iterator<char>(ifstr),
		(istreambuf_iterator<char>()));
	float input[D][D][C] = {
#include "inputs_batch_0"
		//#include "t_im"
	};

	float ref[N] = {
//#include "t_l1"
#include "outputs_batch_0"
		//#include "t_cifar"
	};
	vector<float, aligned_allocator<float> > h_im(D*D*C), h_out(N);
	for(int i=0;i<D;i++)
		for(int j=0;j<D;j++)
			for(int c=0;c<C;c++)
				h_im[i*D*C+j*C+c]=input[i][j][c];
	int err = 0, TT =N;
	float ave = 0;

	try
	{
		vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);

		cl::Context context(CL_DEVICE_TYPE_ACCELERATOR);
		vector<cl::Device> devices=context.getInfo<CL_CONTEXT_DEVICES>();

		cl::Program::Binaries binaries(1, make_pair(programString.c_str(), programString.length()));
		cl::Program program(context,devices,binaries);
		try
		{
			program.build(devices);
		}
		catch (cl::Error err)
		{
			if (err.err() == CL_BUILD_PROGRAM_FAILURE)
			{
				string info;
				program.getBuildInfo(devices[0],CL_PROGRAM_BUILD_LOG, &info);
				cout << info << endl;
				return EXIT_FAILURE;
			}
			else throw err;
		}

		cl::CommandQueue commandQueue(context, devices[0]);

		//		typedef cl::make_kernel<cl::Buffer,cl::Buffer,float,float,float,float,float,float,float,float,float,float,float> kernelType;
		//		kernelType kernelFunctor = kernelType(program, Params::kernel_name);

		cl::Kernel kernel(program,Params::kernel_name);
		auto kernelFunctor = cl::KernelFunctor<cl::Buffer,cl::Buffer>(kernel);


		cl::Buffer d_im(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
				sizeof(float)*D*D*C, h_im.data());
		cl::Buffer d_out(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
				sizeof(float)*N, h_out.data());
		std::vector<cl::Memory> inBufVec, outBufVec;
		inBufVec.push_back(d_im);
		outBufVec.push_back(d_out);
		commandQueue.enqueueMigrateMemObjects(inBufVec,0);
		clock_t start = clock();
		cl::EnqueueArgs enqueueArgs(commandQueue,cl::NDRange(1),cl::NDRange(1));
		cl::Event event = kernelFunctor(enqueueArgs,
				d_im,d_out
				);

		commandQueue.enqueueMigrateMemObjects(outBufVec,CL_MIGRATE_MEM_OBJECT_HOST);
		commandQueue.finish();
		event.wait();

		clock_t t = clock() - start;
		cout << "The execution lasts for "<< (float)t /CLOCKS_PER_SEC <<" s (CPU time)."<<endl;
		for(int k=0;k<N;k++){
			float r = *(ref+k);
			float output = h_out[k];
			if(r == 0.0){
				TT --;
				if(output == 0.0)
					continue;
				else
					err ++;
			}
			float diff = abs(output / r - 1);// ref[i][j][k]);
			ave+=diff;
			if (diff > 1e-1)
				err ++;
			cout	<<k<<','
				<<output<<','
				<<r << ','
				<<endl;
		}
		cout << "there are in total " << err << " errors."<<endl;
		cout << "the ave error is " << ave/TT << " ."<<endl;
	}
	catch (cl::Error err)
	{
		cerr
			<< "Error:\t"
			<< err.what()
			<< endl;

		return EXIT_FAILURE;
	}

	if(err ==0 )
		return EXIT_SUCCESS;
	else
		return EXIT_FAILURE;

}
