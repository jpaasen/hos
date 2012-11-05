#include "PointSource.h"

PointSource::PointSource(void)
{
}

PointSource::~PointSource(void)
{
}


std::vector<Coordinate<float> >* PointSource::getCoord() 
{
	return makeVectorWithElem<Coordinate<float> >(coord);
}

std::vector<float>* PointSource::getFreq() 
{
	return makeVectorWithElem<float>(freq);
}

void PointSource::setFreq(std::vector<float> &freq)
{
	if (freq.size() != 1)
	{
		//throw exception
	} else {
		this->freq = freq[0];
	}
}

std::vector<float>* PointSource::getApod()
{
	return makeVectorWithElem<float>(apodize);
}

void PointSource::setApod(std::vector<float> &apod)
{
	if (apod.size() != 1)
	{
		//throw exception
	} else {
		apodize = apod[0];
	}
}

std::vector<float>* PointSource::getTimeDelay()
{
	return makeVectorWithElem<float>(timeDelay);
}

void PointSource::updateTimeDelay(std::vector<float> &timeDelay)
{
	if (timeDelay.size() != 1)
	{
		//throw exception
	} else {
		this->timeDelay -= timeDelay[0];
		this->timeDelay *= -1.0f;
	}
}

std::vector<float>* PointSource::getPulseLength()
{
	return makeVectorWithElem<float>(pulseLength);
}

void PointSource::setPulseLength(std::vector<float> &pulseLength)
{
	if (pulseLength.size() != 1)
	{
		//throw exception
	} else {
		this->pulseLength = pulseLength[0];
	}
}

std::vector<timestamp>* PointSource::getTimeStamp()
{
	return makeVectorWithElem<timestamp>(timeStamp);
}

void PointSource::setTimeStamp(std::vector<timestamp> &timeStamp)
{
	if (timeStamp.size() != 1)
	{
		//throw exception
	} else {
		this->timeStamp = timeStamp[0];
	}
}

uint PointSource::sizeSrcGroup()
{
	return 1;
}

uint PointSource::sizePointSrc()
{
	return 1;
}

float PointSource::getTimeOfFlight(Coordinate<float> point, float speedOfSound)
{
	Coordinate<float> diff = Coordinate<float>::subtract(point, coord);

	timeDelay = diff.length() / speedOfSound;

	return timeDelay;
}

