#ifndef _CUDA_UTILS_HEADER_
#define _CUDA_UTILS_HEADER_

#include <stdlib.h>
#include <stdio.h>

#include <cuda_runtime.h>

void cuUtilsSafeCall(cudaError err);

void cuUtilSetDevice();

#endif