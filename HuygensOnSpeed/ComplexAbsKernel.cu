/**
* Part of the ...
*
* Huygen's Principle Simulator - Huygen on Speed
*
* Author: Jon Petter Åsen
* E-mail: jon.p.asen@ntnu.no - jp.aasen@gmail.com
**/
#ifndef _COMPLEX_ABS_CUDA_KERNEL_
#define _COMPLEX_ABS_CUDA_KERNEL_

#include "CudaIncludes.h"
#include "Defines.h"

// TODO: This should have been a cuh-file.

/**
* Calcs abs value or phase angle of complex float buffer
*/
__global__ void ComplexAbsKernel(float* absBuffer, 
								 const cuComplex* complexBuffer, 
								 const unsigned int n,
								 const bool envelope)
{

	const unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

	if (index < n)
	{
		if (envelope) {

			absBuffer[index] = cuCabsf(complexBuffer[index]); // display envelope
			// TODO: Just do multiplication with conjugated value and take real part.

		} else { 

			cuComplex value = complexBuffer[index];

			//absBuffer[index] = PI + atan2f(value.y, value.x); // display phase angle [0 2pi]

			absBuffer[index] = 1.0f + sinf(PI + atan2f(value.y, value.x)); // display sin of phase angle [0 2pi]

			//absBuffer[index] = value.x; // display real part -> gives problems mith the use of cublas min/max of abs(.). 

			//absBuffer[index] = value.y; // display imag part -> the same for this
		}
	}
}

#endif