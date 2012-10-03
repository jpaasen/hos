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

#include "CudaUtils.h"

#include <stdlib.h>
#include <stdio.h>

#include <cuda_gl_interop.h>
//#include <cublas.h>
#include <cublas_v2.h>

class DisplayResponse : public IDisplayResponse
{
private:
	Dimension<int> dispDim;		// display size
	float dynamicRange;			// dynamic range of display
	bool envelope;				// display envelope or phase angle
	
	GLuint pbo;					// openGL pixel buffer object (pbo)
	uint *d_pbo_buffer;			// cuda mapping of pbo
	cudaGraphicsResource *g_res; // CUDA 3.0 > OpenGL interoperability resource

	void drawBuffer();
	

public:
	DisplayResponse(Dimension<int> dim, float dynamRange = 45.0f, bool envelope = true);
	~DisplayResponse();

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

	// disp dim functions
	void setDispDim(Dimension<int> dim);
	Dimension<int> getDispDim();
	int getWidth();
	void setWidth(int w);
	int getHeight();
	void setHeight(int h);

	// other functionality
	bool dispEnvelopeIsOn() {
		return envelope;
	}
	void switchDispMode() {
		envelope = !envelope;
	}
	void setDispMode(bool envelope) {
		this->envelope = envelope;
	}

	float getDynamicRange() {
		return dynamicRange;
	}
	void setDynamicRange(float dynamicRange) {
		this->dynamicRange = dynamicRange;
	}
	void updateDynamicRange(float increment) {
		dynamicRange += increment;
		if (dynamicRange < 5.0f) dynamicRange = 5.0f;
	}

	int numOfPixels() {
		return dispDim.w * dispDim.h;
	}
};
