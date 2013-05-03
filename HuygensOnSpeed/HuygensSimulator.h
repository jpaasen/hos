/**
* Part of the ...
*
* Huygen's Principle Simulator - Huygen on Speed
*
* Author: Jon Petter Åsen
* E-mail: jon.p.asen@ntnu.no - jp.aasen@gmail.com
**/
#pragma once

#include <vector>
#include <time.h>

#include "Defines.h"
#include "Coordinate.h"
#include "ObservationArea.h"
#include "IHuygensPrinciple.h"
#include "PaintTool.h"
#include "HuygensCuComplex.h"

class HuygensSimulator 
{

private:

	// propagation variables
	float simSecondsInCPUSeconds;
	float deltaSimSecInCPUSec;
	bool pauseSim;

	// Timestamp variables
	clock_t currentCPUTime;
	clock_t prevCPUTime;
	timestamp currentTime;
	timestamp deltaTime;

	ObservationArea* observationSpace;	// observation space

	IHuygensPrinciple<float, cuComplex>* huygen; // calculator

	std::vector<ISource<float>*> sourceList;	// list of sources

	int totalNumberOfPointSources;

	clock_t cpuTimeBeforeCalculation;
	clock_t cpuTimeAfterCalculation;


public:

	// TODO: move all implementation details into cpp-file.
	// TODO: make a source manager object that can be handed over to other objects so that they can add sources them self.

	// Constructor
	HuygensSimulator(IHuygensPrinciple<float, cuComplex>* huygen) 
	{
		this->huygen = huygen;
		this->observationSpace = NULL;

		// init default values
		simSecondsInCPUSeconds	= 1e4f;
		deltaSimSecInCPUSec		= 10000;
		pauseSim				= false;
		deltaTime				= 5e-7f;//1e-6f;//2e-7f;

		totalNumberOfPointSources = 0;

		// init timing variables
		currentCPUTime = clock();
		prevCPUTime = currentCPUTime;
		currentTime = 0.0f;
	}

	// Destructor
	virtual ~HuygensSimulator()
	{
		if (huygen)
		{
			delete huygen;
			huygen = NULL;
		}
		sourceList.clear();
	}

	// sim loop control functions
	void play()			{pauseSim = false;}
	void pause()		{pauseSim = true;}
	void playPause()	{pauseSim = !pauseSim;}
	void restart()		{currentTime = 0;}
	void stepBack()		{currentTime -= deltaTime;}
	void stepForward()	{currentTime += deltaTime;}
	bool isInPlay()		{return !pauseSim;} 

	void increaseSimSpeed()
	{
		if (simSecondsInCPUSeconds > deltaSimSecInCPUSec) {
			simSecondsInCPUSeconds -= deltaSimSecInCPUSec;
		} else if (simSecondsInCPUSeconds > 0) {
			simSecondsInCPUSeconds -= deltaSimSecInCPUSec/10;
		}
		
	}

	void decreaseSimSpeed()
	{
		if (simSecondsInCPUSeconds >= deltaSimSecInCPUSec) { 
			simSecondsInCPUSeconds += deltaSimSecInCPUSec;
		} else {
			simSecondsInCPUSeconds += deltaSimSecInCPUSec/10;
		}
	}

	//
	// TODO: Method for calculating next simulation result and return result. (Result can be on the GPU or CPU. Need to work that out)
	//cuComplex* calcSimulation(ObservationArea* obsArea);
	void calcSimulation(ObservationArea* obsArea);
	//
	//

	// getters and setters
	float getSpeedOfSound() { 
		return observationSpace->getSpeedOfSound();
	}

	void setSpeedOfSound(float s) 
	{
		observationSpace->setSpeedOfSound(s);
	}

	float getSimCPUTimeFactor() {return simSecondsInCPUSeconds;}

	float getCurrentTime() {return currentTime;}

	std::vector<ISource<float>*>* getSourceList()
	{
		return &sourceList;
	}

	void addSrc(ISource<float>* s)
	{
		sourceList.push_back(s);
		totalNumberOfPointSources += s->sizePointSrc();
	}

	void clearSrcList()
	{
		sourceList.clear();
		totalNumberOfPointSources = 0;
	}

	int numberOfPointSources()
	{
		return totalNumberOfPointSources;
	}

	void updateNumberOfPointSources(int n)
	{
		totalNumberOfPointSources += n;
	}

	void updateSourceFrequencies(const float& f);

	// slow mov / fast mov
	// adjust timestep


	// Converter functions

	// MOVE THESE functions to the GPU-version of the simulator 
	// get float* to list of sources. Format: [x1 x2 ... xn y1 y2 ... yn z1 z2 ... zn]
	// Might change to using float3. Matlab does not support float3 in kernels. 
	void convertSourceList(std::vector<float> &newSourceList) {
		int n = totalNumberOfPointSources;

      newSourceList.resize(3*n);

		uint k = 0;

		for (uint i = 0; i < sourceList.size(); i++)
		{
			std::vector<Coordinate<float> >* sourceCoord = sourceList[i]->getCoord();

			for (uint j = 0; j < sourceCoord->size(); j++)
			{
				newSourceList[k]		= sourceCoord->at(j).x;
				newSourceList[k + n]	= sourceCoord->at(j).y;
				newSourceList[k + n*2]	= sourceCoord->at(j).z;

				k++;
			}

			delete sourceCoord;
		}
	}

	// TODO: Move to the GPU-version of the simulator
	void getFreqList(std::vector<float> &freqList) {

		freqList.resize(totalNumberOfPointSources);

		uint n = (uint) sourceList.size();
		uint k = 0;

		for (uint i = 0; i < n; i++)
		{
			
			std::vector<float>* srcFreq = sourceList[i]->getFreq();

			for (uint j = 0; j < srcFreq->size(); j++)
			{
				freqList[k] = srcFreq->at(j);
				k++;
			}

			delete srcFreq;
		}
	}

	// TODO: Move to the GPU-version of the simulator
	void getApodList(std::vector<float> &apodList) {
		
      apodList.resize(totalNumberOfPointSources);

		uint n = (uint) sourceList.size();
		uint k = 0;

		for (uint i = 0; i < n; i++)
		{
			
			std::vector<float>* srcApod = sourceList[i]->getApod();

			for (uint j = 0; j < srcApod->size(); j++)
			{
				apodList[k] = srcApod->at(j);
				k++;
			}

			delete srcApod;
		}
	}

	// TODO: Move to the GPU-version of the simulator
	void getTimeDelayList(std::vector<float> &timeDelayList) {
      
      timeDelayList.resize(totalNumberOfPointSources);

		uint n = (uint) sourceList.size();
		uint k = 0;

		for (uint i = 0; i < n; i++)
		{
			std::vector<float>* srcTimeDelay = sourceList[i]->getTimeDelay();

			for (uint j = 0; j < srcTimeDelay->size(); j++)
			{
				timeDelayList[k] = srcTimeDelay->at(j);
				k++;
			}
			delete srcTimeDelay;
		}
	}

	// TODO: Move to the GPU-version of the simulator
	void getTimeStampList(std::vector<float> &timeStampList) {
		
      timeStampList.resize(totalNumberOfPointSources);

		uint n = (uint) sourceList.size();
		uint k = 0;

		for (uint i = 0; i < n; i++)
		{
			std::vector<float>* srcTimeStamp = sourceList[i]->getTimeStamp();

			for (uint j = 0; j < srcTimeStamp->size(); j++)
			{
				timeStampList[k] = srcTimeStamp->at(j);
				k++;
			}
			delete srcTimeStamp;
		}
	}

	// TODO: Move to the GPU-version of the simulator
	void getPulseLengthList(std::vector<uint> &pulsLengthList) {
		
      pulsLengthList.resize(totalNumberOfPointSources);

		uint n = (uint) sourceList.size();
		uint k = 0;

		for (uint i = 0; i < n; i++)
		{
			std::vector<float>* srcPulseLength = sourceList[i]->getPulseLength();

			for (uint j = 0; j < srcPulseLength->size(); j++)
			{
				pulsLengthList[k] = uint(floor(srcPulseLength->at(j)));
				k++;
			}
			delete srcPulseLength;
		}
	}


	/**
	* Calc and update time delays for all sources relativ to a common focus point.
	*
	* Source timestamps are neglected.
	*
	* TODO: To get an array with directive elements an array element consisting of multiple sources need to be equaly delayed
	**/
	void updateTimeDelays(Coordinate<float> focusPoint)
	{
		// find source with maximum timeOfFlight
		float maxTimeOfFlight = 0;
		for (uint i = 0; i < sourceList.size(); i++)
		{
			float newTimeOfFlight = sourceList[i]->getTimeOfFlight(focusPoint, observationSpace->getSpeedOfSound());
			//sourceList[i]->timeDelay = newTimeOfFlight;

			if (newTimeOfFlight > maxTimeOfFlight)
			{
				maxTimeOfFlight = newTimeOfFlight;
			}
		}
		// give this source 0 time delay and use its timeOfFlight as ref time

		std::vector<float> maxTime;
		maxTime.push_back(maxTimeOfFlight);

		// for each source
		for (uint i = 0; i < sourceList.size(); i++)
		{
			// calc difference between its timeOfFlight and refTime. 
			//sourceList[i]->timeDelay -= maxTimeOfFlight;
			sourceList[i]->updateTimeDelay(maxTime);
		}
	}

	void listSources()
	{

		uint k = 0;

		for (unsigned int i = 0; i < sourceList.size(); i++) 
		{
			std::vector<Coordinate<float> >* srcCoord	= sourceList[i]->getCoord();
			std::vector<float>* srcTimeStamp = sourceList[i]->getTimeStamp();
				
			for (uint j = 0; j < srcCoord->size(); j++)
			{
				printf("(%.2f, %.2f, %.2f) %.2f us\n", srcCoord->at(k).x, srcCoord->at(k).y, srcCoord->at(k).z, srcTimeStamp->at(k)*1000000);

				k++;
			}

			delete srcCoord;
			delete srcTimeStamp;
		}
	}

	float getFrameRate()
	{
		return 1.0f / ((cpuTimeAfterCalculation - cpuTimeBeforeCalculation) / float(CLOCKS_PER_SEC));
	}

};