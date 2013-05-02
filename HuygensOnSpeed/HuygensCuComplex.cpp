/**
* Part of the ...
*
* Huygen's Principle Simulator - Huygen on Speed
*
* Author: Jon Petter Åsen
* E-mail: jon.p.asen@ntnu.no - jp.aasen@gmail.com
**/
#include "HuygensCuComplex.h"

#if defined(CALC_WITH_CUDA) || defined(DISP_WITH_CUDA)

#else

   cuComplex make_cuComplex(float a, float b) {
      cuComplex s;
      s.x = a; s.y = b;
      return s;
   }

   cuComplex cuCaddf(cuComplex a, cuComplex b) {
      return make_cuComplex(a.x + b.x, a.y + b.y);
   }
#endif