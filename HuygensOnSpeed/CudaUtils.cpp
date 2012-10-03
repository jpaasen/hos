#include "CudaUtils.h"

#include <climits>

void cuUtilsSafeCall(cudaError err)
{
	if (err != cudaSuccess) {
		fprintf(stderr, "%s\n", cudaGetErrorString(err));
	}
}

void cuUtilSetDevice()
{
	int dev_count;
	cudaGetDeviceCount(&dev_count);

	int min_dev_sm = INT_MAX;
	int max_dev_sm = 0;
	int min_dev = 0;
	int max_dev = 0;

	cudaDeviceProp dev_prop;
	int i;
	for (i = 0; i < dev_count; ++i) {
		cudaGetDeviceProperties(&dev_prop, i);

		if (dev_prop.multiProcessorCount > max_dev_sm) {
			max_dev_sm = dev_prop.multiProcessorCount;
			max_dev = i;
		} else if (dev_prop.multiProcessorCount < min_dev_sm) {
			min_dev_sm = dev_prop.multiProcessorCount;
			min_dev = i;
		}
	}
	
	cudaError e = cudaSetDevice(max_dev);

	if (e != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(e));
	}
}