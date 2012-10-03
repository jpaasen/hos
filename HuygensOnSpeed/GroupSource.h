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
#include "PointSource.h"

/**
* Models a directive element by using multiple point sources.
* TODO: Change name to DirectiveSource 
*/ 
class GroupSource :
	public ISource<float>
{
private:
	std::vector<ISource*> sourceList;

public:
	GroupSource(void);
	virtual ~GroupSource(void);

	GroupSource(
		std::vector<Coordinate<float>> coord, 
		std::vector<float> freq, 
		std::vector<float> apod, 
		std::vector<float> timeDelay,
		std::vector<float> pulseLength, 
		std::vector<float> ts)
	{
		for (uint i = 0; i < coord.size(); i++)
		{
			sourceList.push_back(new PointSource());
		}
	}
	
	GroupSource(
		std::vector<Coordinate<float>> coord, 
		float freq, 
		float apod, 
		float timeDelay, 
		float pulseLength, 
		float timeStamp)  
	{
		for (uint i = 0; i < coord.size(); i++)
		{
			sourceList.push_back(new PointSource());
		}
	}

	std::vector<Coordinate<float>>* getCoord();

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

	uint sizeSrcGroup();
	uint sizePointSrc();

	float getTimeOfFlight(Coordinate<float> point, float speedOfSound);
};
