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
#include "HuygensCuComplex.h"

#include <GL/glew.h>

/**
* Interface for mapping result to display
**/
class IDisplayResponse
{
protected:
   Dimension<int> dispDim; // display size
   float dynamicRange;		// dynamic range of display
   bool envelope;				// display envelope or phase angle

public:
   IDisplayResponse(Dimension<int> dim, float dynamRange, bool envelope) : dispDim(dim), dynamicRange(dynamRange), envelope(envelope) {}
	virtual ~IDisplayResponse() {}
   
   virtual void mapResponseToDisplay(
      const cuComplex* result,	// result
      const uint obsW,			   // observation space width
      const uint obsH,			   // observation space height	
      const bool resultOnGPU	   // true if result is in GPU memory 
		) = 0;
   
   // pixel buffer object (pbo) functions
   virtual GLuint createPBO() = 0;
   virtual GLuint getPBO() = 0;
	virtual void deletePBO() = 0;

   // display size functions
   void setDispDim(Dimension<int> dim) {
      dispDim = dim;
      createPBO();
   }

   Dimension<int> getDispDim() {
      return dispDim;
   }

   int getWidth() {
      return dispDim.w;
   }

   void setWidth(int w) {
      dispDim.w = w; 
      createPBO();
   }

   int getHeight() {
      return dispDim.h;
   }

   void setHeight(int h) {
      dispDim.h = h;
      createPBO();
   }

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