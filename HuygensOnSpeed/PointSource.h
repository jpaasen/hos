/**
* Part of the ...
*
* Huygen's Principle Simulator - Huygen on Speed
*
* Author: Jon Petter Åsen
* E-mail: jon.p.asen@ntnu.no - jp.aasen@gmail.com
**/
#pragma once
#include "ISource.h"

#include "Defines.h"
#include "Coordinate.h"

#include <math.h>

class PointSource :
	public ISource<float>
{
private:
	Coordinate<float> coord;
	float freq;
	float apodize;
	float timeDelay;
	float pulseLength; // 0 == inf
	timestamp timeStamp;

	template <typename E>
	std::vector<E>* makeVectorWithElem(E elem)
	{
		std::vector<E>* vec = new std::vector<E>();
		vec->push_back(elem);
		return vec;
	}

public:
	PointSource(void);
	virtual ~PointSource(void);

	PointSource(Coordinate<float> c, float f, float a, float t, float pL, float ts) 
		: coord(c), freq(f), apodize(a), timeDelay(t), pulseLength(pL), timeStamp(ts) {}

	std::vector<Coordinate<float> >* getCoord();

	std::vector<float>* getFreq();
	void setFreq(std::vector<float> &freq);

	std::vector<float>* getApod();
	void setApod(std::vector<float> &apod);

	std::vector<float>* getTimeDelay();
	void updateTimeDelay(std::vector<float> &timeDelay);

	std::vector<float>* getPulseLength();
	void setPulseLength(std::vector<float> &pulseLength);

	std::vector<timestamp>* getTimeStamp();
	void setTimeStamp(std::vector<timestamp> &timeStamp);

	unsigned int sizeSrcGroup();
	unsigned int sizePointSrc();

	float getTimeOfFlight(Coordinate<float> point, float speedOfSound);
};
