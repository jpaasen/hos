#include "ObservationArea.h"	

#include <stdexcept>

Coordinate<unsigned int> ObservationArea::nObsPoints() {
	if (numObsPoints == 0) // number of obs points has not been calculated before (or it is 0)
	{
		// any how, we then do the calculations
		Coordinate<float> diff = Coordinate<float>::subtract(maxLimits, minLimits);
		diff.mul(resolution);

		unsigned int nX = static_cast<unsigned int>(floor(diff.x));
		unsigned int nY = static_cast<unsigned int>(floor(diff.y));
		unsigned int nZ = static_cast<unsigned int>(floor(diff.z));

		if (nX == 0) nX = 1; // make one point if interval is zero
		if (nY == 0) nY = 1; 
		if (nZ == 0) nZ = 1; 

		numObsPointsVec = Coordinate<unsigned int>(nX, nY, nZ);

		numObsPoints = numObsPointsVec.reduceMul();
	}

	return numObsPointsVec;
}

Coordinate<float> ObservationArea::areaSize() {
	return Coordinate<float>::subtract(maxLimits, minLimits);
}

unsigned int ObservationArea::numelObsPoints() {
	if (numObsPoints == 0)
	{
		Coordinate<unsigned int> coord = nObsPoints();
		numObsPoints = coord.reduceMul();
	}

	return numObsPoints;
}

float* ObservationArea::getObsPoints() {
   if (obsPointsGPU.empty()) 
	{
		Coordinate<unsigned int> n = nObsPoints();

		unsigned int numObs = n.reduceMul();

      obsPointsGPU.resize(numObs * 3);

		for (unsigned int i = 0; i < numObs; i++) 
		{
			obsPointsGPU[i]				= obsPoints[i].x;
			obsPointsGPU[i + numObs]	= obsPoints[i].y;
			obsPointsGPU[i + numObs*2]	= obsPoints[i].z;
		}
	}

	return obsPointsGPU.data();
}

Coordinate<float> ObservationArea::getPosition(unsigned int x, unsigned int z, unsigned int w, unsigned int h) {
	Coordinate<float> t = Coordinate<float>(x/float(w), 0, z/float(h));
	return Coordinate<float>::elemlerp(maxLimits, minLimits, t);
}

void ObservationArea::createObsGrid() {

	Coordinate<unsigned int> n = nObsPoints();
   obsPoints.resize(n.reduceMul());

	float x = minLimits.x;
	float y;
	float z;

	float increment = 1 / resolution;

	for (unsigned int i = 0; i < n.x; i++) {

		y = minLimits.y;

		for (unsigned int j = 0; j < n.y; j++) {

			z = minLimits.z;

			for (unsigned int k = 0; k < n.z; k++) {

				Coordinate<float> coord(x,y,z);
				obsPoints[i*n.y*n.z + j*n.z + k] = coord;

				z += increment;
			}
			y += increment;
		}
		x += increment;
	}
}

ObservationArea::ObservationArea(int dim, 
								 Coordinate<float> minL, 
								 Coordinate<float> maxL, 
								 float resolution,
								 float speedOfSound,
								 bool resultOnGPU) 
								 : dim(dim), minLimits(minL), maxLimits(maxL), resolution(resolution), speedOfSound(speedOfSound), resultOnGPU(resultOnGPU) {
#if defined(DISP_WITH_CPU)
   if (resultOnGPU) {
      throw std::runtime_error("ObservationArea can not be located on the GPU if DisplayResponseCPU is used.");
   }
#endif
   
   d_res_gpu = NULL;

	numObsPoints = 0;

	createObsGrid();

	createResMem();
}

ObservationArea::~ObservationArea() {
	if (d_res_gpu) deleteResMem();
}

void ObservationArea::deleteResMem() {
#if defined(DISP_WITH_CUDA)
   if (resultOnGPU) {
      if (d_res_gpu) {
         cuUtilsSafeCall( cudaFree(d_res_gpu) );
         d_res_gpu = NULL;
      }
   }
#endif
}

void ObservationArea::createResMem() {
	if (d_res_gpu) {
		deleteResMem();
	}
	if (resultOnGPU) {
#if defined(DISP_WITH_CUDA)
		cuUtilsSafeCall( cudaMalloc<cuComplex>(&d_res_gpu, sizeof(cuComplex)*numelObsPoints()) );
#endif
	} else {
      d_res.resize(numelObsPoints());
   }
}

cuComplex* ObservationArea::getResMem() {
   if (resultOnGPU) {
      return d_res_gpu;
   } else {
	   return d_res.data();
   }
}