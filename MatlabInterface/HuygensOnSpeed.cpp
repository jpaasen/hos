
#include <math.h>
#include <matrix.h>
#include <mex.h>

#include <string>
#include <complex>

//#include <cuComplex.h>
#include "../HuygensOnSpeed/HuygensOnGPU.h"
#include "../HuygensOnSpeed/HuygensOnCPU.h"

template <typename T, typename K>
void copyKArrayToTArray(T* to, const K* from, size_t N)
{
	for (size_t i = 0; i < N; ++i)
	{
		to[i] = (T) from[i];

		//mexPrintf("%f ", to[i]);

		//if (i % 10 == 0) mexPrintf("\n");
	}
	//mexPrintf("\n");
}

/*void copyCuComplexArrayToMxArray(mxArray* to, const cuComplex* from, size_t N)
{
	for (size_t i = 0; i < N; ++i)
	{
		to[i] = std::complex<double>(from[i].x, from[i].y);
	}
}*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	/* check for proper number of arguments */
	if(nrhs != 10) {
		std::string howToUse = std::string("10 inputs required. \n") +  
			std::string("coordObs,\t\t\tObservation coordiantes: dim3\n") +
			std::string("coordSrc,\t\t\tSource coordinates: dim3\n") +
			std::string("fSrc,\t\t\t\tFrequency: dim1 length=1 value>0\n") +
			std::string("apodSrc,\t\t\tApodization: dim1 length=length(coordSrc)\n") +
			std::string("steerFocusDelaySrc,\tSteer-focus delays: dim1 length=length(coordSrc)\n") +
			std::string("srcTimeStamp,\t\tTime stamp telling when source starts to fire: dim1 length=length(coordSrc)\n") +
			std::string("srcPulseLength,\t\tPulse length 0 == Inf: dim1 length=length(coordSrc)\n") +
			std::string("timestampObs,\t\tCurrent timestamp for this observation: dim1 length=1\n") + 	
			std::string("c0,\t\t\t\t\tPropagation speed: dim1 length=1\n") + 
			std::string("useGPUSimulator: GPU (true), CPU (false)");
		mexErrMsgIdAndTxt("HuygensOnSpeed:nrhs", howToUse.c_str());
	}
	if(nlhs!=1) {
		mexErrMsgIdAndTxt("HuygensOnSpeed:nlhs",
			"One output required.");
	}

	// huygens simulator
	HuygensOnGPU huygenGPU;
	HuygensOnCPU huygenCPU;

	// copy obs coords from mxArray to float array
	const mwSize* obsDims = mxGetDimensions(prhs[0]);
	const double* coordObsD = mxGetPr(prhs[0]);
	const uint nObs = obsDims[0];
	
#ifdef _DEBUG
	mexPrintf("Size of prhs[0]: [%d, %d]\n", nObs, obsDims[1]);
#endif

	float* coordObs = (float*) mxCalloc(nObs*obsDims[1], sizeof(float));
	copyKArrayToTArray<float, double>(coordObs, coordObsD, nObs*obsDims[1]);

	// allocate memory for the resulting field
	cuComplex* d_res = (cuComplex*) mxCalloc(nObs, sizeof(cuComplex));

	// copy src coords from mxArray to float array
	const mwSize* srcDims = mxGetDimensions(prhs[1]);
	const uint nSrc = srcDims[0];
	double* coordSrcD = mxGetPr(prhs[1]);

#ifdef _DEBUG
	mexPrintf("Size of prhs[1]: [%d, %d]\n", nSrc, srcDims[1]);
#endif

	float* coordSrc = (float*) mxCalloc(nSrc*srcDims[1], sizeof(float));
	copyKArrayToTArray<float, double>(coordSrc, coordSrcD, nSrc*srcDims[1]);	
	
	// copy frequencies 
	float* fSrc = (float*) mxCalloc(nSrc, sizeof(float));
	double* fSrcD = mxGetPr(prhs[2]);
	copyKArrayToTArray<float, double>(fSrc, fSrcD, nSrc);

	// copy apodization
	float* apodSrc = (float*) mxCalloc(nSrc, sizeof(float));
	double* apodSrcD = mxGetPr(prhs[3]);
	copyKArrayToTArray<float, double>(apodSrc, apodSrcD, nSrc);

	float* steerFocusDelaySrc = (float*) mxCalloc(nSrc, sizeof(float));
	double* steerFocusDelaySrcD = mxGetPr(prhs[4]);
	copyKArrayToTArray<float, double>(steerFocusDelaySrc, steerFocusDelaySrcD, nSrc);

	float* srcTimeStamp = (float*) mxCalloc(nSrc, sizeof(float));
	double* srcTimeStampD = mxGetPr(prhs[5]);
	copyKArrayToTArray<float, double>(srcTimeStamp, srcTimeStampD, nSrc);

	uint* srcPulseLength = (uint*) mxCalloc(nSrc, sizeof(uint));
	int* srcPulseLengthI = (int*) mxGetPr(prhs[6]);
	copyKArrayToTArray<uint, int>(srcPulseLength, srcPulseLengthI, nSrc);

	const float timestampObs = (float) mxGetScalar(prhs[7]);
	const float refTime = 0.0f;
	const float c0 = (float) mxGetScalar(prhs[8]);
	const bool useGPUSimulator = (bool) mxGetScalar(prhs[9]);
	const bool resultOnGPU = false;

#ifdef _DEBUG
	mexPrintf("%f %f %f %d\n", timestampObs, refTime, c0, resultOnGPU);
	mexPrintf("%d %d %d %d %d %d %d\n", sizeof(uint), sizeof(int), sizeof(short), sizeof(float), sizeof(double), sizeof(cuComplex), sizeof(size_t));
#endif

	if (useGPUSimulator) {
		mexPrintf("hold on, Huygens is running on Speed\n");
		huygenGPU.calcFieldResponse(
			d_res,
			nObs, coordObs,		// Observation # and coordiantes
			nSrc, coordSrc,		// Source #, coordinates,
			fSrc, apodSrc,		// frequencies, apodization
			steerFocusDelaySrc,	// and steer-focus delays
			srcTimeStamp,		// time stamp telling when source starts to fire
			srcPulseLength,		// pulse length 0 == Inf
			timestampObs,		// Current timestamp for this observation
			refTime,			// Reference time for calculating attenuation 	
			c0,
			resultOnGPU);		// true if d_res is on the GPU
	} else {
		mexPrintf("stay put, Huygens is grumpy\n");
		huygenCPU.calcFieldResponse(
			d_res,
			nObs, coordObs,		// Observation # and coordiantes
			nSrc, coordSrc,		// Source #, coordinates,
			fSrc, apodSrc,		// frequencies, apodization
			steerFocusDelaySrc,	// and steer-focus delays
			srcTimeStamp,		// time stamp telling when source starts to fire
			srcPulseLength,		// pulse length 0 == Inf
			timestampObs,		// Current timestamp for this observation
			refTime,			// Reference time for calculating attenuation 	
			c0,
			resultOnGPU);		// true if d_res is on the GPU
	}
	//associate outputs
	mxArray* out;
	out = plhs[0] = mxCreateDoubleMatrix(nObs, 1, mxCOMPLEX);

	//copy d_res too out
	double* re = mxGetPr(out);
	double* im = mxGetPi(out);
	for (size_t i = 0; i < nObs; ++i)
	{
		re[i] = (double) d_res[i].x;
		im[i] = (double) d_res[i].y;
	}

	mxFree((void*)d_res);
	mxFree((void*)coordObs);
	mxFree((void*)coordSrc);
	mxFree((void*)apodSrc);
	mxFree((void*)steerFocusDelaySrc);
	mxFree((void*)srcTimeStamp);
	mxFree((void*)srcPulseLength);

}