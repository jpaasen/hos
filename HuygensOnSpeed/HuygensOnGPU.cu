#include "HuygensOnGPU.h"

#include "HuygensKernel.cu"

cuComplex* HuygensOnGPU::calcFieldResponse(
	ObservationArea* obsArea,
	std::vector<ISource<float>*> &src,
	const float timestampObs)
{
	return 0;
}

void HuygensOnGPU::calcFieldResponse(cuComplex* d_res,
									 const unsigned int nObs, const float* coordObs,	// Observation # and coordiantes
									 const unsigned int nSrc, const float* coordSrc,	// Source #, coordinates,
									 const float* fSrc, const float* apodSrc,	// frequencies, apodization
									 const float* steerFocusDelaySrc,			// and steer-focus delays
									 const float* srcTimeStamp,					// time stamp telling when source starts to fire
									 const unsigned int* srcPulseLength,				// pulse length 0 == Inf
									 const float timestampObs,					// Current timestamp for this observation
									 const float refTime,						// Reference time for calculating attenuation 	
									 const float c0,
									 const bool resultOnGPU)					// true if d_res is on the GPU
{

#ifdef USE_CONST_MEMORY // Give this def to preprocessor for const memory. Use constant memory for devices of cc < 2.0  
	if (nSrc > CONST_MEM_SIZE) {
		fprintf(stderr, "Maximum number of sources is restricted to %d when constant memory is used\n", CONST_MEM_SIZE);
	} 
	else
	{
#endif

		//cuUtilSetDevice(); // select device (max sm count)

		// copy points etc which is not in device memory to the device
		float* d_coordObs = NULL;
		unsigned int memSize = sizeof(float) * nObs * 3;
		cuUtilsSafeCall( cudaMalloc<float>(&d_coordObs, memSize) );
		cuUtilsSafeCall( cudaMemcpy(d_coordObs, coordObs, memSize, cudaMemcpyHostToDevice) );

		float* d_coordSrc = NULL;
		memSize = sizeof(float) * nSrc * 3;
		cuUtilsSafeCall( cudaMalloc<float>(&d_coordSrc, memSize) );
		cuUtilsSafeCall( cudaMemcpy(d_coordSrc, coordSrc, memSize, cudaMemcpyHostToDevice) );

		float* d_fSrc = NULL;
		memSize = sizeof(float) * nSrc;

#ifdef USE_CONST_MEMORY
		cuUtilsSafeCall( cudaMemcpyToSymbol(c_fSrc, fSrc, memSize) );
#else
		cuUtilsSafeCall( cudaMalloc<float>(&d_fSrc, memSize) );
		cuUtilsSafeCall( cudaMemcpy(d_fSrc, fSrc, memSize, cudaMemcpyHostToDevice) );
#endif	

		float* d_apodSrc = NULL;
#ifdef USE_CONST_MEMORY
		cuUtilsSafeCall( cudaMemcpyToSymbol(c_apodSrc, apodSrc, memSize) );
#else
		cuUtilsSafeCall( cudaMalloc<float>(&d_apodSrc, memSize) );
		cuUtilsSafeCall( cudaMemcpy(d_apodSrc, apodSrc, memSize, cudaMemcpyHostToDevice) );
#endif
		

		float* d_steerFocusDelaySrc = NULL;
#ifdef USE_CONST_MEMORY
		cuUtilsSafeCall( cudaMemcpyToSymbol(c_steerFocusDelaySrc, steerFocusDelaySrc, memSize) );
#else
		cuUtilsSafeCall( cudaMalloc<float>(&d_steerFocusDelaySrc, memSize) );
		cuUtilsSafeCall( cudaMemcpy(d_steerFocusDelaySrc, steerFocusDelaySrc, memSize, cudaMemcpyHostToDevice) );
#endif

		float* d_srcTimeStamp = NULL;
#ifdef USE_CONST_MEMORY
		cuUtilsSafeCall( cudaMemcpyToSymbol(c_srcTimeStamp, srcTimeStamp, memSize) );
#else
		cuUtilsSafeCall( cudaMalloc<float>(&d_srcTimeStamp, memSize) );
		cuUtilsSafeCall( cudaMemcpy(d_srcTimeStamp, srcTimeStamp, memSize, cudaMemcpyHostToDevice) );
#endif

		unsigned int* d_srcPulseLength = NULL;
#ifdef USE_CONST_MEMORY
		cuUtilsSafeCall( cudaMemcpyToSymbol(c_srcPulseLength, srcPulseLength, nSrc * sizeof(unsigned int)) );	
#else	
		cuUtilsSafeCall( cudaMalloc<unsigned int>(&d_srcPulseLength, nSrc * sizeof(unsigned int)) );
		cuUtilsSafeCall( cudaMemcpy(d_srcPulseLength, srcPulseLength, nSrc * sizeof(unsigned int), cudaMemcpyHostToDevice) );
#endif

		cuComplex* d_res2;
		if (!resultOnGPU) {
			cuUtilsSafeCall( cudaMalloc<cuComplex>(&d_res2, nObs * sizeof(cuComplex)) );
		}

		// Create block grid
		dim3 block(HOS_BLOCK_SIZE,1,1);
		dim3 grid((nObs - 1)/block.x + 1, 1);

		// launch kernel
		HuygensKernel<<<grid,block>>>(
			(resultOnGPU? d_res : d_res2), 
			nObs, d_coordObs, nSrc, d_coordSrc,
			d_fSrc, d_apodSrc, d_steerFocusDelaySrc, d_srcTimeStamp, d_srcPulseLength,
			timestampObs, refTime, c0);

		// check for errors
		cudaThreadSynchronize();

		cuUtilsSafeCall( cudaGetLastError() );

		if (!resultOnGPU) {
			cuUtilsSafeCall( cudaMemcpy(d_res, d_res2, nObs * sizeof(cuComplex), cudaMemcpyDeviceToHost) );
			cuUtilsSafeCall( cudaFree(d_res2) );	
		}

		// free device memory
		cuUtilsSafeCall( cudaFree(d_apodSrc) );
		cuUtilsSafeCall( cudaFree(d_coordObs) );
		cuUtilsSafeCall( cudaFree(d_coordSrc) );
		cuUtilsSafeCall( cudaFree(d_fSrc) );
		cuUtilsSafeCall( cudaFree(d_steerFocusDelaySrc) );
		cuUtilsSafeCall( cudaFree(d_srcTimeStamp) );
		cuUtilsSafeCall( cudaFree(d_srcPulseLength) );

#ifdef USE_CONST_MEMORY
	}
#endif

	//free((void *)apodSrc);
	//free((void *)coordObs); // this one is now cleaned up by the observation object
	//free((void *)coordSrc);
	//free((void *)fSrc);
	//free((void *)steerFocusDelaySrc);
	//free((void *)srcTimeStamp);
	//free((void *)srcPulseLength);
}