/**
* Part of the ...
*
* Huygen's Principle Simulator - Huygen on Speed
*
* Author: Jon Petter Åsen
* E-mail: jon.p.asen@ntnu.no - jp.aasen@gmail.com
**/
#pragma once

#include "Defines.h"
#include "HuygensCuComplex.h"
#include "ObservationArea.h"
#include "ISource.h"

#include <vector>

template <typename T, typename complex>
class IHuygensPrinciple
{
public:

	virtual ~IHuygensPrinciple() {};

	// TODO: Change all float arrays to a single source array. Lett the implementation take care of copying it to the right memory (hence for a GPU)
	virtual void calcFieldResponse(
		cuComplex* d_res,								// result // TODO: Remove dependency on cuComplex, can be tricky!
		const uint nObs, const T* coordObs,			// Observation # and coordiantes
		const uint nSrc, const T* coordSrc,			// Source #, coordinates,
		const T* fSrc, const T* apodSrc,			// frequencies, apodization
		const T* steerFocusDelaySrc,				// and steer-focus delays
		const T* srcTimeStamp,						// time stamp telling when source starts to fire
		const uint* srcPulseLength,					// pulse length 0 == Inf
		const T timestampObs,						// Current timestamp for this observation
		const T refTime,							// Reference time for calculating attenuation (not in use)
		const T c0,									// Speed of sound
		const bool resultOnGPU)						// true if d_res is GPU memory
		= 0; // pure virtual function

	// new function, taking objects instead of lists
	virtual complex* calcFieldResponse(
		ObservationArea* obsArea,
		std::vector<ISource<T>*> &src,
		const T timestampObs)
		= 0;
};