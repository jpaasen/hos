/**
* Part of the ...
*
* Huygen's Principle Simulator - Huygen on Speed
*
* Author: Jon Petter Åsen
* E-mail: jon.p.asen@ntnu.no - jp.aasen@gmail.com
**/
#pragma once

#include "stdlib.h"
#include "Defines.h"
#include "Coordinate.h"

#include <math.h>

/**
* Wave source / center for a Green's function
*
* Legacy implementation. Use PointSource and the ISource interface instead.
**/
struct Source
{
	Coordinate<float> coord;
	float freq;
	float apodize;
	float timeDelay;
	uint pulseLenght; // 0 == inf
	timestamp timeStamp;

	Source(Coordinate<float> c, float f, float a, float t, uint pL, float ts) 
		: coord(c), freq(f), apodize(a), timeDelay(t), pulseLenght(pL), timeStamp(ts) {}

	~Source() {}

	// Get time of flight from this source to given Coordinate point
	float getTimeOfFlight(Coordinate<float> point, float speedOfSound);
};