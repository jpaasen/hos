/**
* Part of the ...
*
* Huygen's Principle Simulator - Huygens on Speed
*
* Author: Jon Petter Åsen
* E-mail: jon.p.asen@ntnu.no - jp.aasen@gmail.com
**/
#ifndef _HUYGENS_CUDA_KERNEL_
#define _HUYGENS_CUDA_KERNEL_

// define only one at once
//#define USE_CONST_MEMORY
#define USE_SHARED_MEMORY 

#include "CudaIncludes.h"
#include "Defines.h"

#ifdef USE_CONST_MEMORY // Use constant memory for devices of cc < 2.0. Give this def to preprocessor for const memory. 
#define CONST_MEM_SIZE 3000
__constant__ float c_fSrc[CONST_MEM_SIZE];					// frequencies
__constant__ float c_apodSrc[CONST_MEM_SIZE];				// apodization
__constant__ float c_steerFocusDelaySrc[CONST_MEM_SIZE];	// steer-focus delays
__constant__ float c_srcTimeStamp[CONST_MEM_SIZE];			// time stamp telling when source starts to fire
__constant__ unsigned int  c_srcPulseLength[CONST_MEM_SIZE];		// length of pulse in PW mode 
#endif

/**
*
* Input:
*			- Source list
*			- Observation points x,y,z
*			- Resolution of observation area
*			- Size of observation area
*			- Size (in int units) of observation area
*			- Speed of sound (c)
*			- Apodize
*			- Focus and steering delays
*			- Frequency (list?)
*			- Observation timestamp
*			- Reference time
*			- jomega (complex number) (2*pi*frequency)

*			(Additional for PW)
*			- Pulse length (list?)
*
* Output:
*			- Respons in observation area
*
* Input format:
*
*			For maximum memory coalescing and matlab complient code, 
*			coordinates are saved as:
*			[x0 x1 ... xn y0 y1 ... yn z0 z1 ... zn]
*
*			Result is saved with linear indexing.
*
**/
__global__ void HuygensKernel(cuComplex* resp,							// Output respons
							  const unsigned int nObs, const float* coordObs,	// Observation # and coordiantes
							  const unsigned int nSrc, const float* coordSrc,	// Source #, coordinates,
							  const float* fSrc, const float* apodSrc,	// frequencies, apodization
							  const float* steerFocusDelaySrc,			// and steer-focus delays
							  const float* srcTimeStamp,				// time stamp telling when source starts to fire
							  const unsigned int* srcPulseLength,				// pulse length for PW-mode (0 == Inf)
							  const float timestampObs,					// Current timestamp for this observation
							  const float refTime,						// Reference time for calculating attenuation (not in use)
							  const float c0)							// Speed of sound

							  // TODO: Add azimut and elevation index of sources for ultrasim for correct apodization
{

#ifdef USE_SHARED_MEMORY
#define SRC_BUFFER_SIZE 192//256 // this must equal the block size
	__shared__ float3 srcs[SRC_BUFFER_SIZE];
	__shared__ float  fSrcs[SRC_BUFFER_SIZE];
	__shared__ float  delaySrcs[SRC_BUFFER_SIZE];
	__shared__ float  timeStampSrcs[SRC_BUFFER_SIZE];
	__shared__ unsigned int   pulseLengthSrcs[SRC_BUFFER_SIZE];
	__shared__ float  apodSrcs[SRC_BUFFER_SIZE];
#endif

	// calc linear index of observation point
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < nObs)
	{
		// current observation point
		float3 obs = make_float3(coordObs, index, nObs);

		// init respons
		cuComplex respTemp = make_cuComplex(0.0f, 0.0f);

#ifdef USE_SHARED_MEMORY
		unsigned int nGroups = 0;
		if (nSrc > 0) {
			nGroups = (nSrc - 1) / SRC_BUFFER_SIZE + 1;
		}
		for (unsigned int group = 0; group < nGroups; group++)
		{

			unsigned int n = group * SRC_BUFFER_SIZE + threadIdx.x;
			unsigned int nSrcIngroup = SRC_BUFFER_SIZE;

			if (nSrc < SRC_BUFFER_SIZE + group * SRC_BUFFER_SIZE) {
				nSrcIngroup = nSrc - group * SRC_BUFFER_SIZE;
			}

			__syncthreads();

			if (n < nSrc) {
				srcs[threadIdx.x]				= make_float3(coordSrc, n, nSrc);
				fSrcs[threadIdx.x]				= fSrc[n];
				delaySrcs[threadIdx.x]			= steerFocusDelaySrc[n];
				timeStampSrcs[threadIdx.x]		= srcTimeStamp[n];
				pulseLengthSrcs[threadIdx.x]	= srcPulseLength[n];
				apodSrcs[threadIdx.x]			= apodSrc[n];
			}

			__syncthreads();

			for (unsigned int srcIdx = 0; srcIdx < nSrcIngroup; srcIdx++) {

				float3 src = srcs[srcIdx];

#else

		// loop over all source points
		for (unsigned int n = 0; n < nSrc; n++) 
		{
			// Optimalization plans:
			// all threads will read this value! TODO: Check if this value gets broadcasted. Info: Broadcasting only works for shared memory!
			// If not -> TODO: Delegate one read into shared memory to each thread. If nSrc > blockDim.x, deligate multiple reads to each thread.
			// For CUDA 2.0: Shared memory == User-managed L2 cache. Broadcasting might however help improving the memory throughput.
			float3 src = make_float3(coordSrc, n, nSrc);
#endif

#ifndef USE_SHARED_MEMORY
#ifdef USE_CONST_MEMORY
			// fetch steer-focus delay, src timestamp, frequency and pulse length from constant memory
			float tStFo			= c_steerFocusDelaySrc[n];	// TODO: read one time for each block into shared memory (The same issue as explained for src coordinate reads)
			float timestampSrc	= c_srcTimeStamp[n];		// TODO: read one time for each block into shared memory
			float frequencySrc	= c_fSrc[n];				// TODO: read one time for each block into shared memory
			unsigned int pulseL			= c_srcPulseLength[n];		// TODO: read one time for each block into shared memory
			float apod			= c_apodSrc[n];				// TODO: read one time for each block into shared memor
#else
			// fetch steer-focus delay, src timestamp, frequency and pulse length from global memory
			float tStFo			= steerFocusDelaySrc[n];	
			float timestampSrc	= srcTimeStamp[n];			
			float frequencySrc	= fSrc[n];					
			unsigned int pulseL			= srcPulseLength[n];
			float apod			= apodSrc[n];
#endif
#else
			float tStFo			= delaySrcs[srcIdx];	
			float timestampSrc	= timeStampSrcs[srcIdx];			
			float frequencySrc	= fSrcs[srcIdx];					
			unsigned int  pulseL		= pulseLengthSrcs[srcIdx];
			float apod			= apodSrcs[srcIdx];
#endif

			// time of flight from source to current observation point
			float dist = absf(subf(obs,src));

			// clamp dist to prevent huge amplitude values
			float minDist = c0/(2*frequencySrc); 
			dist = fmaxf(minDist, dist); 

			float currentFlightTime = timestampObs - timestampSrc - tStFo;

			float timeOFligth = dist / c0;

			if (pulseL == 0) // cw
			{
				if (currentFlightTime > timeOFligth) // source is alive in this obs point
				{
					// find actual time for response computation
					float t = timeOFligth - currentFlightTime;
					//float t = currentFlightTime - timeOFligth;

					// calc attenuation
					float att = dist;

					float p = 2.0f * PI * frequencySrc * t;	// complex phase of greens function
					float r = apod / att;					// complex modulus of greens function

					// calc respons of source in current obs point
					float cosptr, sinptr;  
					sincosf(p, &sinptr, &cosptr);
					cuComplex newResp = make_cuComplex(r*cosptr, r*sinptr);

					respTemp = cuCaddf(respTemp, newResp); // add to respTemp
				} else
					if (timeOFligth <= 1/frequencySrc) // hack to get a relative bright spot in pw-mode. Preventing depth normalization from happening.
					{
						float cosWeightFac = cosf(PI * timeOFligth / (2/frequencySrc));
						respTemp = cuCaddf(respTemp, make_cuComplex(cosWeightFac * cosWeightFac / dist, 0));
					}
			} 
			else // pw
			{
				float pulseLInSec = pulseL / frequencySrc;
				float halfPulseLInSec = pulseLInSec / 2.0f;

				if (currentFlightTime > timeOFligth - halfPulseLInSec && currentFlightTime < timeOFligth + halfPulseLInSec)
				{
					// pw source is alive in this observation point
					float t = timeOFligth - currentFlightTime;

					// calc attenuation
					float att = dist;

					float cosWeightPuls = cosf(PI * t / pulseLInSec);

					float p = 2.0f * PI * frequencySrc * t;	// complex phase of greens function
					float r = cosWeightPuls * cosWeightPuls * apod / att;	// complex modulus of greens function

					// calc respons of source in current obs point
					float cosptr, sinptr;  
					sincosf(p, &sinptr, &cosptr);
					cuComplex newResp = make_cuComplex(r*cosptr, r*sinptr);

					respTemp = cuCaddf(respTemp, newResp); // add to respTemp

				} else {

					if (timeOFligth <= 1/frequencySrc) // hack to get a relative bright spot in pw-mode. Preventing depth normalization from happening.
					{
						float cosWeightFac = cosf(PI * timeOFligth / (2/frequencySrc));
						respTemp = cuCaddf(respTemp, make_cuComplex(cosWeightFac * cosWeightFac / dist, 0));
					}
				}
			}
		}

#ifdef USE_SHARED_MEMORY
		}
#endif

		resp[index] = respTemp; // save respons for this observation point
	}
}

#endif