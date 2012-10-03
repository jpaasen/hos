/**
* Part of the ...
*
* Huygen's Principle Simulator - Huygen on Speed
*
* Author: Jon Petter Åsen
* E-mail: jon.p.asen@ntnu.no - jp.aasen@gmail.com
**/
#pragma once

#include "Defines.h"
#include "Dimension.h"
#include "cuComplex.h"

#include <GL/glew.h>

/**
* Interface for mapping result to display
**/
class IDisplayResponse
{
public:

	virtual ~IDisplayResponse() {};

	virtual void mapResponseToDisplay(
		const cuComplex* result,	// result
		const uint obsW,			// observation space width
		const uint obsH,			// observation space height	
		const bool resultOnGPU		// true if result is in GPU memory 
		) = 0;

	// pixel buffer object (pbo) functions
	virtual GLuint createPBO() = 0;
	virtual GLuint getPBO() = 0;
	virtual void deletePBO() = 0;

	// display dim functions
	virtual void setDispDim(Dimension<int> dim) = 0;
	virtual Dimension<int> getDispDim() = 0;
	virtual int getWidth() = 0;
	virtual void setWidth(int w) = 0;
	virtual int getHeight() = 0;
	virtual void setHeight(int h) = 0;

	// other functionality

	// visualize envelope or phase angle
	virtual bool dispEnvelopeIsOn() = 0;
	virtual void switchDispMode() = 0;
	virtual void setDispMode(bool envelope) = 0;

	// dynamic range of display
	virtual float getDynamicRange() = 0;
	virtual void setDynamicRange(float dynamicRange) = 0;
	virtual void updateDynamicRange(float increment) = 0;

	virtual int numOfPixels() = 0;
};