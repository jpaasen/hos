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

//#include <stdlib.h>
#include <stdio.h>

#include <cuda_gl_interop.h>
//#include <cublas.h>
#include <cublas_v2.h>

class DisplayResponse : public IDisplayResponse
{
private:
   GLuint pbo;					// openGL pixel buffer object (pbo)
   unsigned int *d_pbo_buffer;			// cuda mapping of pbo
   cudaGraphicsResource *g_res; // CUDA 3.0 > OpenGL interoperability resource

   void drawBuffer();

public:
   DisplayResponse(Dimension<int> dim, float dynamRange = 45.0f, bool envelope = true);
   ~DisplayResponse();

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
