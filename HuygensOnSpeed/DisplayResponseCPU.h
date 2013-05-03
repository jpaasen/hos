/**
* Part of the ...
*
* Huygen's Principle Simulator - Huygen on Speed
*
* Author: Jon Petter Åsen
* E-mail: jon.p.asen@ntnu.no - jp.aasen@gmail.com
**/
#pragma once

#include "IDisplayResponse.h"
#include <vector>

class DisplayResponseCPU : public IDisplayResponse
{
private:

   std::vector<unsigned int> pbo;

   void drawBuffer(unsigned int* buffer, const unsigned int obsW, const unsigned int obsH);
   void takeAbs(float* absValues, const cuComplex* values, unsigned int N, bool envelope);
   float findMax(float* values, unsigned int N);
   float findMin(float* values, unsigned int N);
   void normalize(unsigned int* output, float* values, float minValue, float maxValue, unsigned int obsW, unsigned int obsH);

public:
   DisplayResponseCPU(Dimension<int> dim, float dynamRange = 45.0f, bool envelope = true);
	~DisplayResponseCPU();

	void mapResponseToDisplay(
		const cuComplex* result, 
		const unsigned int obsW, 
		const unsigned int obsH,
		const bool resultOnGPU	
		);

	// pbo functions
	GLuint createPBO();
	GLuint getPBO();
	void deletePBO();
};