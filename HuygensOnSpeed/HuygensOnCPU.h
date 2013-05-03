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
#include <cmath>
#include "IHuygensPrinciple.h"

class HuygensOnCPU : public IHuygensPrinciple<float, cuComplex>
{
public:
	HuygensOnCPU(void);
	~HuygensOnCPU(void);

	void calcFieldResponse(cuComplex* d_res,
		const unsigned int nObs, const float* coordObs,		// Observation # and coordiantes
		const unsigned int nSrc, const float* coordSrc,		// Source #, coordinates,
		const float* fSrc, const float* apodSrc,	// frequencies, apodization
		const float* steerFocusDelaySrc,			// and steer-focus delays
		const float* srcTimeStamp,					// time stamp telling when source starts to fire
		const unsigned int* srcPulseLength,					// pulse length 0 == Inf
		const float timestampObs,					// Current timestamp for this observation
		const float refTime,						// Reference time for calculating attenuation 	
		const float c0,								// Speed of sound
		const bool resultOnGPU);					// true if d_res is on the GPU

	cuComplex* calcFieldResponse(
		ObservationArea* obsArea,
		std::vector<ISource<float>*> &src,
		const float timestampObs);
};
