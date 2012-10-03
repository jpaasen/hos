/**
* Part of the ...
*
* Huygen's Principle Simulator - Huygen on Speed
*
* Author: Jon Petter Åsen
* E-mail: jon.p.asen@ntnu.no - jp.aasen@gmail.com
**/
#include "IHuygensPrinciple.h"

#include <stdlib.h>
#include <stdio.h>

#include "CudaUtils.h"

#define HOS_BLOCK_SIZE 192

class HuygensOnGPU : public IHuygensPrinciple<float, cuComplex>
{
private:
	cuComplex result;

public:
	HuygensOnGPU() {}
	~HuygensOnGPU() {}

	void calcFieldResponse(cuComplex* d_res,
		const uint nObs, const float* coordObs,		// Observation # and coordiantes
		const uint nSrc, const float* coordSrc,		// Source #, coordinates,
		const float* fSrc, const float* apodSrc,	// frequencies, apodization
		const float* steerFocusDelaySrc,			// and steer-focus delays
		const float* srcTimeStamp,					// time stamp telling when source starts to fire
		const uint* srcPulseLength,					// pulse length 0 == Inf
		const float timestampObs,					// Current timestamp for this observation
		const float refTime,						// Reference time for calculating attenuation 	
		const float c0,
		const bool resultOnGPU);					// true if d_res is on the GPU

	cuComplex* calcFieldResponse(
		ObservationArea* obsArea,
		std::vector<ISource<float>*> &src,
		const float timestampObs);
};