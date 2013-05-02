#include "DisplayResponseCPU.h"

#include <stdlib.h>
#include <stdexcept>
#include <limits>
#include <cmath>

/*void checkOpenGLError() {
   GLenum error = glGetError();
   const GLubyte *errString;
   if (error != GL_NO_ERROR) {
      errString = gluErrorString(error);
      fprintf(stderr, "OpenGL error: %s\n", errString);
   } else {
      printf("OpenGL... OK\n");
   }
}*/

// Constructor
DisplayResponseCPU::DisplayResponseCPU(Dimension<int> dim, float dynamRange, bool envelope) : IDisplayResponse(dim, dynamRange, envelope) 
{
   d_pbo_buffer = 0;
}

// Destructor
DisplayResponseCPU::~DisplayResponseCPU() 
{
	deletePBO();
}

float DisplayResponseCPU::findMin(float* values, uint N) {
   float min_value = std::numeric_limits<float>::max();
   for (uint i = 0; i < N; i++) {
      if (values[i] < min_value) {
         min_value = values[i];
      }
   }
   return min_value;
}

float DisplayResponseCPU::findMax(float* values, uint N) {
   float max_value = std::numeric_limits<float>::min();
   for (uint i = 0; i < N; i++) {
      if (values[i] > max_value) {
         max_value = values[i];
      }
   }
   return max_value;
}

void DisplayResponseCPU::takeAbs(float* absValues, const cuComplex* values, uint N, bool envelope) {

   for (uint i = 0; i < N; i++) {

      cuComplex v = values[i];

      if (envelope) {
         absValues[i] = sqrt(v.x*v.x + v.y*v.y);
      } else {
         absValues[i] = 1.0f + sin(PI + atan2(v.y, v.x));
      }
   }
}

void DisplayResponseCPU::normalize(uint* output, float* values, float minValue, float maxValue, uint obsW, uint obsH) {

   for (uint x = 0; x < obsH; x++) {
      for (uint y = 0; y < obsW; y++) { 

         float normValue = (values[y*obsH + x] - minValue) / (maxValue - minValue);

         float dBValue = 20.0f * log10(normValue); // possible issue with value == 0.

         const uint colorHigh = 0xff;
         const uint colorLow = 0x00;

         float t = 1.0f + dBValue/dynamicRange;
         if (t > 1.0f) t = 1.0f;
         if (t < 0.0f) t = 0.0f;

         const uint color = (uint((1.0f - t)*colorLow) + uint(t*colorHigh));

         //			 alpha			   blue			  green		  red
         output[x*obsW + y] = 0xff000000 | (color << 16) | (color << 8) | color;
      }
   }
}

void DisplayResponseCPU::drawBuffer(uint* buffer, const uint obsW, const uint obsH)
{
	// display result
	// Clear the color part of both back buffers
	glClear(GL_COLOR_BUFFER_BIT);

	// Draw image from right and left PBO
	glDisable(GL_DEPTH_TEST);
	glRasterPos2i(0, 0);

   glDrawBuffer(GL_BACK);

   gluScaleImage(GL_RGBA, obsW, obsH, GL_UNSIGNED_BYTE, buffer, getWidth(), getHeight(), GL_UNSIGNED_BYTE, d_pbo_buffer);
   glDrawPixels(getWidth(), getHeight(), GL_RGBA, GL_UNSIGNED_BYTE, d_pbo_buffer);

   glFinish();
}

/**
* Map complex result to openGL pixel buffer object (PBO).
*
* Before copying to pbo, result is normalized 
*
**/
void DisplayResponseCPU::mapResponseToDisplay(const cuComplex* result, const uint obsW, const uint obsH, const bool resultOnGPU) 
{

	if (resultOnGPU) {
      throw std::runtime_error("DisplayResponsCPU does not support result located on the GPU.");
   }

	const uint n = obsW * obsH;

   // Take abs of result
   float* abs_buffer = (float*) malloc(n*sizeof(float));
   takeAbs(abs_buffer, result, n, envelope);

	// Find maximum and minimum
   float max_value = findMax(abs_buffer, n);
   float min_value = findMin(abs_buffer, n);

   // Normalize
   uint* buffer_norm  = (uint*) malloc(n*sizeof(uint));
   normalize(buffer_norm, abs_buffer, min_value, max_value, obsW, obsH);

   drawBuffer(buffer_norm, obsW, obsH);
   
   free(abs_buffer);
   free(buffer_norm);
}

GLuint DisplayResponseCPU::createPBO()
{
	GLsizeiptr memSize = numOfPixels() * sizeof(GLubyte) * 4;
   d_pbo_buffer = (uint*) malloc(memSize);

	return 0;
}

GLuint DisplayResponseCPU::getPBO()
{
	return 0;
}

void DisplayResponseCPU::deletePBO()
{
   if (d_pbo_buffer != 0) { 
      free(d_pbo_buffer);
   }
}

