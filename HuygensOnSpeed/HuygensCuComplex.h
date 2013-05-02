/**
* Part of the ...
*
* Huygen's Principle Simulator - Huygen on Speed
*
* Author: Jon Petter Åsen
* E-mail: jon.p.asen@ntnu.no - jp.aasen@gmail.com
**/
#pragma once

#if defined(CALC_WITH_CUDA) || defined(DISP_WITH_CUDA)
#include <cuComplex.h>
#else
   struct cuComplex {
      float x;
      float y;  
   };

   cuComplex make_cuComplex(float a, float b);

   cuComplex cuCaddf(cuComplex a, cuComplex b);
#endif