/**
* Part of the ...
*
* Huygen's Principle Simulator - Huygen on Speed
*
* Author: Jon Petter Åsen
* E-mail: jon.p.asen@ntnu.no - jp.aasen@gmail.com
**/
#ifndef _DISPLAY_CUDA_KERNEL_
#define _DISPLAY_CUDA_KERNEL_

#include "CudaIncludes.h"
#include "Defines.h"

// TODO: This should have been a cuh-file.

texture<float, 2, cudaReadModeElementType> response; // texture used for mapping result to display

__constant__ float maxValue;
__constant__ float minValue;

/**
*	Map complex 2D result to openGL pbo 
*
*	Field response needs to be binded to the texture (response) above
*	before launch.
*
**/
__global__ void DisplayKernel(uint* pbo, const uint w, const uint h, const float dynamicRange)
{
	uint x = blockDim.x*blockIdx.x + threadIdx.x;
	uint y = blockDim.y*blockIdx.y + threadIdx.y;

	if (x < h && y < w)
	{
		float value = tex2D(response, x/float(h), y/float(w));

		//const float dynamicRange = 70.0f;
		const uint index = x*w + y; // will specify how result is mapped to the pbo

		// map value to [0 1];
		float normValue = (value - minValue) / (maxValue - minValue);

		float dBValue = 20.0f * log10f(normValue); // possible issue with value == 0.

		const uint colorHigh = 0xff;
		const uint colorLow = 0x00;

		float t = 1.0f + dBValue/dynamicRange;
		if (t > 1.0f) t = 1.0f;
		if (t < 0.0f) t = 0.0f;

		const uint color = (uint((1.0f - t)*colorLow) + uint(t*colorHigh));

		//			 alpha			   blue			  green		  red
		pbo[index] = 0xff000000 | (color << 16) | (color << 8) | color;
	}
}

#endif 