#ifndef _CUDA_INCLUDES_HEADER_
#define _CUDA_INCLUDES_HEADER_

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math_functions.h>

#include "Defines.h"

// Helper function for calculating length of vector
__device__ inline float absf(float3 a)
{ 
	return sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
}

// Helper function for subtracting two vectors
__device__ inline float3 subf(float3 a, float3 b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// extract float3 from float array with stridded access of w
__device__ inline float3 make_float3(const float* a, uint &xIdx, const uint w)
{
	//	     			 xIdx	   yIdx		   zIdx
	return make_float3(a[xIdx], a[xIdx+w], a[xIdx+2*w]);
}

#endif