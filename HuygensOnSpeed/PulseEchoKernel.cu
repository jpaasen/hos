/**
* Part of the ...
*
* Huygen's Principle Simulator - Huygens on Speed
*
* Author: Jon Petter Åsen
* E-mail: jon.p.asen@ntnu.no - jp.aasen@gmail.com
**/
#ifndef _PULSE_ECHO_CUDA_KERNEL_
#define _PULSE_ECHO_CUDA_KERNEL_

#include "CudaIncludes.h"
/**
// Output respons
// Scatterer # and coordiantes
// Scatterer amplitude
// Rx Source #, coordinates, (nRxSrc == 1)
// Tx Source #, coordinates,
// Tx apodization
// Tx steer and focus delays
// Time stamp telling when source starts to fire
// Pulse length for PW-mode (0 == Inf)
// Tx frequency
// Current timestamp for this observation
**/
__global__ void PulseEchoKernel(float* resp,								
								const unsigned int nScat, 
								const float* coordScat,
								const float* ampScat,
								const unsigned int nRx,  const float* coordRx,
								const unsigned int nTx,  const float* coordTx,
								const float* apodSrc,
								const float* steerFocusDelaySrc,
								const float* srcTimeStamp,
								const float srcPulseLength,
								const float timestampObs,
								const float fSrc,
								const float c0)
{

	uint index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < nScat)
	{

		uint zero = 0;

		float3 scat = make_float3(coordScat, index, nScat);
		float3 rx   = make_float3(coordRx, zero, nRx);

		float distRx = absf(subf(scat, rx));

		float respTemp = 0.0f;

		for (uint n = 0; n < nTx; ++n)
		{
			float3 tx = make_float3(coordTx, n, nTx);

			float distTx = absf(subf(scat, tx));

			float tStFo			= steerFocusDelaySrc[n];	
			float timestampSrc	= srcTimeStamp[n];								
			float apod			= apodSrc[n];

			float totalDist    = distRx + distTx;
			float timeOfFlight = totalDist / c0;

			float pulseTOF = timestampObs - timestampSrc - tStFo; // how long since pulse left tx element

			if (srcPulseLength > 0) // PW
			{

				float pulseLInSec = srcPulseLength / fSrc;
				float halfPulseLInSec = pulseLInSec / 2.0f;

				if (pulseTOF > timeOfFlight - halfPulseLInSec && pulseTOF < timeOfFlight + halfPulseLInSec)
				{ // current Rx element is recieving signal from Tx, reflected by scatterer in obs for the given timestampObs

					float t = timeOfFlight - pulseTOF;

					float cosWeightPuls = cosf(PI * t / pulseLInSec); // pulse is made up by a cos^2(-pi/2:pi/2) window
					cosWeightPuls *= cosWeightPuls;

					respTemp += ( (apod * cosWeightPuls / totalDist) * sinf(2.0f * PI * fSrc * t) ); // and a f0 frequency sin function

				}

			} else {} // CW

			float amp = ampScat[index];
			resp[index] = amp * respTemp;
		}
	}
}

#endif