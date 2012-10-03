#include "Source.h"

float Source::getTimeOfFlight(Coordinate<float> point, float speedOfSound)
{
	Coordinate<float> diff = Coordinate<float>::subtract(point, coord);

	return diff.length() / speedOfSound;
}