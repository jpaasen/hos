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
#include "Coordinate.h"
//#include "CudaIncludes.h"
#include "CudaUtils.h"
#include <cuComplex.h>

#include <stdlib.h>
#include <math.h>
#include <stdio.h>

/**
* Represent a discrete grid of field observation points.
**/
class ObservationArea 
{
private:
	int dim;						// space dimension
	Coordinate<float> minLimits;	// [minX minY minZ] extent in m
	Coordinate<float> maxLimits;	// [maxX maxY maxZ] extent in m
	float resolution;				// sampling points per m
	Coordinate<float>* obsPoints;	// list of observation points
	float speedOfSound;

	Coordinate<uint> numObsPointsVec;	// holding the number of obsPoints in x, y and z so that it is calculated only once for this object
	uint numObsPoints;					// in the same way this holdes the total number of observation points

	float* obsPointsGPU;			// list of observation points suited for GPU kernel (coalesced memory reads)

	cuComplex* d_res;				// memory to hold result
	bool resultOnGPU;				// true if result are allocated in GPU memory

	void createObsGrid();

public:
	/**
	*	dim = dimension of observation area
	*	min = [minX minY minZ] extent in meter
	*	max = [maxX maxY maxZ] extent in meter
	*	resolution = number of samples per meter
	*
	*	From this the constructor will calculate a grid of observation points
	**/
	ObservationArea(int dim, Coordinate<float> minL, Coordinate<float> maxL, float resolution, float speedOfSound, bool resultOnGPU);

	~ObservationArea();

	/** 
	*	Return a list of observation points suited for GPU transfer: 
	*	The memory returned will be cleaned up by this object.
	*	[x1 ... xn y1 ... yn z1 ... zn]
	**/
	float* getObsPoints();

	/** Return number of observation points in x, y and z**/
	Coordinate<unsigned int> nObsPoints();

	/** Area size in meters in x, y and z**/
	Coordinate<float> areaSize();				// as coordinate/vector
	float areaSizeX() {return areaSize().x;}	// as scalars
	float areaSizeY() {return areaSize().y;}
	float areaSizeZ() {return areaSize().z;}

	/** Number of observation points **/
	unsigned int numelObsPoints();

	/** Map (x,z) to coordinate in observation area wher x in [0 w-1] and z in [0 h-1]**/
	Coordinate<float> getPosition(uint x, uint z, uint w, uint h);

	/** Getters and setters **/
	Coordinate<float> getMinLim() {return minLimits;}
	Coordinate<float> getMaxLim() {return maxLimits;}

	float getSpeedOfSound() {return speedOfSound;}
	void setSpeedOfSound(float sos) 
	{
		if (sos > 0)
		{
			this->speedOfSound = sos;
		}
	}

	void deleteResMem();
	void createResMem();
	cuComplex* getResMem();

	bool resultIsOnGPU() {return resultOnGPU;}
};