#include "HuygensOnCPU.h"

HuygensOnCPU::HuygensOnCPU(void)
{
}

HuygensOnCPU::~HuygensOnCPU(void)
{
}

cuComplex* HuygensOnCPU::calcFieldResponse(
	ObservationArea* obsArea,
	std::vector<ISource<float>*> &src,
	const float timestampObs)
{
	return 0;
}

void HuygensOnCPU::calcFieldResponse(cuComplex *d_res, 
									 const unsigned int nObs, const float *coordObs, 
									 const unsigned int nSrc, const float *coordSrc, 
									 const float *fSrc, const float *apodSrc, 
									 const float *steerFocusDelaySrc, 
									 const float *srcTimeStamp, 
									 const unsigned int *srcPulseLength, 
									 const float timestampObs, 
									 const float refTime, 
									 const float c0,
									 const bool resultOnGPU) 
{
	// calc linear index of observation point

	cuComplex *resp = (cuComplex*) malloc(sizeof(cuComplex)*nObs);

   #pragma omp parallel for
	for (int index = 0; index < nObs; index++)
	{

		if (index < nObs)
		{
			// current observation point
			float3 obs = make_float3(coordObs, index, nObs);

			// init respons
			cuComplex respTemp = make_cuComplex(0.0f, 0.0f);

			// loop over all source points
			for (int n = 0; n < nSrc; n++) 
			{
				// Optimalization plans:
				// all threads will read this value! TODO: Check if this value gets broadcasted. Info: Broadcasting only works for shared memory!
				// If not -> TODO: Delegate one read into shared memory to each thread. If nSrc > blockDim.x, deligate multiple reads to each thread.
				// For CUDA 2.0: Shared memory == User-managed L2 cache. Broadcasting might however help improving the memory throughput.
				float3 src = make_float3(coordSrc, n, nSrc);  

				// fetch steer-focus delay, src timestamp, frequency and pulse length from global memory
				float tStFo			= steerFocusDelaySrc[n];	
				float timestampSrc	= srcTimeStamp[n];			
				float frequencySrc	= fSrc[n];					
				uint pulseL			= srcPulseLength[n];
				float apod			= apodSrc[n];

				// time of flight from source to current observation point
				float dist = absf(subf(obs,src));

				// clamp dist to prevent huge amplitude values
				float minDist = c0/(2*frequencySrc); 
				dist = (minDist < dist)? dist : minDist; 

				float timeOFligth = dist / c0;

				float currentFlightTime = timestampObs - timestampSrc - tStFo;

				if (pulseL == 0) // cw
				{
					if (currentFlightTime > timeOFligth) // source is alive in this obs point
					{
						// find actual time for response computation
						float t = timeOFligth - currentFlightTime;
						//float t = currentFlightTime - timeOFligth;

						// calc attenuation
						float att = dist;

						float r = apod / att;				// complex modulus of greens function
						float p = 2.0f * PI * frequencySrc * t;	// complex phase of greens function

						// calc respons of source in current obs point
						float cosptr = cos(p); 
						float sinptr = sin(p);  
						cuComplex newResp = make_cuComplex(r*cosptr, r*sinptr);

						respTemp = cuCaddf(respTemp, newResp); // add to respTemp
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
						cosWeightPuls *= cosWeightPuls;

						float r = cosWeightPuls * apod / att;	// complex modulus of greens function
						float p = 2.0f * PI * frequencySrc * t;		// complex phase of greens function

						// calc respons of source in current obs point
						float cosptr = cos(p);
						float sinptr = sin(p);
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
			resp[index] = respTemp; // save respons for this observation point
		}
	}

	// copy calculated field to the GPU for presentation
	if (resultOnGPU) {
		cudaMemcpy(d_res, resp, sizeof(cuComplex)*nObs, cudaMemcpyHostToDevice);
	} else {
		memcpy(d_res, resp, sizeof(cuComplex)*nObs);
	}

	free(resp);

	//free((void *)apodSrc);
	//free((void *)coordObs); // this one is now cleaned up by the observation object
	//free((void *)coordSrc);
	//free((void *)fSrc);
	//free((void *)steerFocusDelaySrc);
	//free((void *)srcTimeStamp);
	//free((void *)srcPulseLength);
}
