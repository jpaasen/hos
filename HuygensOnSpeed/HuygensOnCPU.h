/**
* Part of the ...
*
* Huygen's Principle Simulator - Huygen on Speed
*
* Author: Jon Petter Åsen
* E-mail: jon.p.asen@ntnu.no - jp.aasen@gmail.com
**/
#pragma once

#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <vector_functions.h>
#include <cmath>
#include "IHuygensPrinciple.h"

class HuygensOnCPU : public IHuygensPrinciple<float, cuComplex>
{
public:
	HuygensOnCPU(void);
	~HuygensOnCPU(void);

	void calcFieldResponse(cuComplex* d_res,
		const uint nObs, const float* coordObs,		// Observation # and coordiantes
		const uint nSrc, const float* coordSrc,		// Source #, coordinates,
		const float* fSrc, const float* apodSrc,	// frequencies, apodization
		const float* steerFocusDelaySrc,			// and steer-focus delays
		const float* srcTimeStamp,					// time stamp telling when source starts to fire
		const uint* srcPulseLength,					// pulse length 0 == Inf
		const float timestampObs,					// Current timestamp for this observation
		const float refTime,						// Reference time for calculating attenuation 	
		const float c0,								// Speed of sound
		const bool resultOnGPU);					// true if d_res is on the GPU

	cuComplex* calcFieldResponse(
		ObservationArea* obsArea,
		std::vector<ISource<float>*> &src,
		const float timestampObs);

	// Helper function for calculating length of vector
	float absf(float3 a)
	{ 
		return sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
	}

	// Helper function for subtracting two vectors
	float3 subf(float3 a, float3 b)
	{
		return ::make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
	}

	// extract float3 from float array with stridded access of w
	float3 make_float3(const float* a, int &xIdx, const int w)
	{
		//	     			 xIdx	   yIdx		   zIdx
		return ::make_float3(a[xIdx], a[xIdx+w], a[xIdx+2*w]);
	}
};
