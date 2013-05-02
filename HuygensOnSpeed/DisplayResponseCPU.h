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

class DisplayResponseCPU : public IDisplayResponse
{
private:

   uint *d_pbo_buffer;

   void drawBuffer(uint* buffer, const uint obsW, const uint obsH);
   void takeAbs(float* absValues, const cuComplex* values, uint N, bool envelope);
   float findMax(float* values, uint N);
   float findMin(float* values, uint N);
   void normalize(uint* output, float* values, float minValue, float maxValue, uint obsW, uint obsH);

public:
   DisplayResponseCPU(Dimension<int> dim, float dynamRange = 45.0f, bool envelope = true);
	~DisplayResponseCPU();

	void mapResponseToDisplay(
		const cuComplex* result, 
		const uint obsW, 
		const uint obsH,
		const bool resultOnGPU	
		);

	// pbo functions
	GLuint createPBO();
	GLuint getPBO();
	void deletePBO();
};