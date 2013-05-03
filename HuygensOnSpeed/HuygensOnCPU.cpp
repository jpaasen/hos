#include "HuygensOnCPU.h"

#if defined(DISP_WITH_CUDA)
   #include <cuda_runtime.h>
   #include <vector_functions.h>
#else
   struct float3 {
      float x, y, z;
   };

   float3 make_float3(float a, float b, float c) {
      float3 s;
      s.x = a; s.y = b; s.z = c;
      return s;
   }
#endif

// Helper function for calculating length of vector
float absf(float3 a) { 
   return sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
}

// Helper function for subtracting two vectors
float3 subf(float3 a, float3 b) {
   return ::make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

   // extract float3 from float array with stridded access of w
float3 make_float3(const float* a, unsigned int &xIdx, const int w) {
         //	     			 xIdx	   yIdx		   zIdx
   return ::make_float3(a[xIdx], a[xIdx+w], a[xIdx+2*w]);
}

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
	for (unsigned int index = 0; index < nObs; index++)
	{

		if (index < nObs)
		{
			// current observation point
			float3 obs = make_float3(coordObs, index, nObs);

			// init respons
			cuComplex respTemp = make_cuComplex(0.0f, 0.0f);

			// loop over all source points
			for (unsigned int n = 0; n < nSrc; n++) 
			{
				float3 src = make_float3(coordSrc, n, nSrc);  

				// fetch steer-focus delay, src timestamp, frequency and pulse length from global memory
				float tStFo			= steerFocusDelaySrc[n];	
				float timestampSrc	= srcTimeStamp[n];			
				float frequencySrc	= fSrc[n];					
				unsigned int pulseL			= srcPulseLength[n];
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
#if defined(DISP_WITH_CUDA)
		cudaMemcpy(d_res, resp, sizeof(cuComplex)*nObs, cudaMemcpyHostToDevice);
#endif
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
