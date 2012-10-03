#include "HuygensSimulator.h"

/*HuygensSimulator::HuygensSimulator()
{

}*/

/*HuygensSimulator::~HuygensSimulator()
{

}*/

/*cuComplex* HuygensSimulator::calcSimulation(ObservationArea *obsArea)
{	
	return huygen->calcFieldResponse(obsArea, sourceList, currentTime);
}*/

void HuygensSimulator::calcSimulation(ObservationArea *obsArea)
{

	clock_t newCPUTime = currentCPUTime;

	if (!pauseSim) {
		newCPUTime = clock();
	}

	if (newCPUTime == currentCPUTime) 
	{
		currentCPUTime = prevCPUTime;
	}
	// match with cpu time. 1 ms equals 1 cpu second. 
	// Increment if deltaTime in cpu time has pased since last update

	float deltaCPUTimeInSimTime = float(newCPUTime - currentCPUTime) / float(simSecondsInCPUSeconds*CLOCKS_PER_SEC);

	// TODO: it is now only updating at 10 us per CPU second. There must not this kind of constraint!
	if (deltaCPUTimeInSimTime > deltaTime)	{

		if (!pauseSim) 
		{
			stepForward();
		}

		prevCPUTime = currentCPUTime;
		currentCPUTime = newCPUTime;


		observationSpace = obsArea;

		cpuTimeBeforeCalculation = clock();

		float* coordSrc = convertSourceList();
		float* fSrc = getFreqList();
		float* apodSrc = getApodList();
		float* steerFocusDelaySrc = getTimeDelayList();
		float* srcTimeStamp = getTimeStampList();
		uint* srcPulseLength = getPulseLengthList();

		huygen->calcFieldResponse(
			observationSpace->getResMem(),
			observationSpace->numelObsPoints(), observationSpace->getObsPoints(),
			sourceList.size(), coordSrc, 
			fSrc, apodSrc, steerFocusDelaySrc, srcTimeStamp, srcPulseLength,
			this->currentTime, this->currentTime, 
			observationSpace->getSpeedOfSound(), observationSpace->resultIsOnGPU());

		cpuTimeAfterCalculation = clock();

		free((void *)coordSrc);
		//free((void *)coordObs); // this one is now cleaned up by the observation object
		free((void *)fSrc);
		free((void *)apodSrc);
		free((void *)steerFocusDelaySrc);
		free((void *)srcTimeStamp);
		free((void *)srcPulseLength);
	}
}

void HuygensSimulator::updateSourceFrequencies(float& const f)
{

	size_t s = sourceList.size();
	std::vector<float> fVector;
	fVector.push_back(f);

	for (size_t i = 0; i < s; ++i) {
		sourceList[i]->setFreq(fVector);
	}
}