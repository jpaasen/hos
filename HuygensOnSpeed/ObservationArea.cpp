#include "ObservationArea.h"	

Coordinate<uint> ObservationArea::nObsPoints() 
{
	if (numObsPoints == 0) // number of obs points has not been calculated before (or it is 0)
	{
		// any how, we then do the calculations
		Coordinate<float> diff = Coordinate<float>::subtract(maxLimits, minLimits);
		diff.mul(resolution);

		uint nX = uint(floor(diff.x));
		uint nY = uint(floor(diff.y));
		uint nZ = uint(floor(diff.z));

		if (nX == 0) nX = 1; // make one point if interval is zero
		if (nY == 0) nY = 1; 
		if (nZ == 0) nZ = 1; 

		numObsPointsVec = Coordinate<uint>(nX, nY, nZ);

		numObsPoints = numObsPointsVec.reduceMul();
	}

	return numObsPointsVec;
}

Coordinate<float> ObservationArea::areaSize()
{
	return Coordinate<float>::subtract(maxLimits, minLimits);
}

uint ObservationArea::numelObsPoints()
{
	if (numObsPoints == 0)
	{
		Coordinate<uint> coord = nObsPoints();
		numObsPoints = coord.reduceMul();
	}

	return numObsPoints;
}

float* ObservationArea::getObsPoints() 
{
	if (obsPointsGPU == NULL) 
	{
		Coordinate<uint> n = nObsPoints();

		uint numObs = n.reduceMul();

		obsPointsGPU = (float*) malloc(numObs * sizeof(float) * 3);

		for (uint i = 0; i < numObs; i++) 
		{
			obsPointsGPU[i]				= obsPoints[i].x;
			obsPointsGPU[i + numObs]	= obsPoints[i].y;
			obsPointsGPU[i + numObs*2]	= obsPoints[i].z;
		}
	}

	return obsPointsGPU;
}

Coordinate<float> ObservationArea::getPosition(uint x, uint z, uint w, uint h) 
{
	Coordinate<float> t = Coordinate<float>(x/float(w), 0, z/float(h));
	return Coordinate<float>::elemlerp(maxLimits, minLimits, t);
}

void ObservationArea::createObsGrid()
{
	Coordinate<uint> n = nObsPoints();

	size_t memSize = n.x * n.y * n.z * sizeof(Coordinate<float>);

#ifdef _DEBUG
	printf("Obs memsize: %d MB\n", memSize/1000000);
#endif

	obsPoints = (Coordinate<float>*) malloc(memSize);

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
								 : dim(dim), minLimits(minL), maxLimits(maxL), resolution(resolution), speedOfSound(speedOfSound), resultOnGPU(resultOnGPU) 
{
	obsPoints = NULL;
	obsPointsGPU = NULL;
	d_res = NULL;

	numObsPoints = 0;

	createObsGrid();

	createResMem();
}

ObservationArea::~ObservationArea() 
{
	if (obsPoints) free(obsPoints);
	if (obsPointsGPU) free(obsPointsGPU);
	if (d_res) deleteResMem();
}

void ObservationArea::deleteResMem()
{
	if (d_res)
	{
		if (resultOnGPU) {
			cudaFree(d_res);
		} else {
			free(d_res);
		}
		d_res = NULL;
	}
}

void ObservationArea::createResMem() 
{
	if (d_res) {
		deleteResMem();
	}
	if (resultOnGPU) {
		cuUtilsSafeCall( cudaMalloc<cuComplex>(&d_res, sizeof(cuComplex)*numelObsPoints()) );
	} else {
		d_res = (cuComplex*) malloc(sizeof(cuComplex)*numelObsPoints());
	}
}

cuComplex* ObservationArea::getResMem()
{
	return d_res;
}