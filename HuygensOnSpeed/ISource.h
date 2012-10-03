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

#include <vector>

template<typename T>
class ISource 
{
public:

	// (*) Length of vector must equal sizeSrcGroup(), otherwise the first element is used on all sources in the group

	virtual ~ISource() {};

	virtual std::vector<Coordinate<T>>* getCoord() = 0;

	virtual std::vector<T>* getFreq() = 0;
	virtual void setFreq(std::vector<T> &freq) = 0; // (*)

	virtual std::vector<T>* getApod() = 0;
	virtual void setApod(std::vector<T> &apod) = 0; // (*)

	virtual std::vector<T>* getTimeDelay() = 0;
	virtual void updateTimeDelay(std::vector<T> &timeDelay) = 0; // (*) // 

	virtual std::vector<T>* getPulseLength() = 0; // 0 == inf, 
	virtual void setPulseLength(std::vector<T> &pulseLength) = 0;	// (*)

	virtual std::vector<timestamp>* getTimeStamp() = 0;
	virtual void setTimeStamp(std::vector<timestamp> &timeStamp) = 0;

	virtual uint sizeSrcGroup() = 0; // return number of sources on next level if the source has sub-sources
	virtual uint sizePointSrc() = 0; // return total number of point sources in this source

	// Get time of flight from the center of this source given a point and a the medium's speed of sound
	// After call, time of fligh can also be retrived by calling getTimeDelay.
	virtual T getTimeOfFlight(Coordinate<float> point, float speedOfSound) = 0;  
};