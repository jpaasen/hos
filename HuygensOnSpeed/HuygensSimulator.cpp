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

void HuygensSimulator::calcSimulation(ObservationArea *obsArea) {

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

      std::vector<float> coordSrc;
      convertSourceList(coordSrc);
		std::vector<float> fSrc;
      getFreqList(fSrc);
		std::vector<float> apodSrc;
      getApodList(apodSrc);
		std::vector<float> steerFocusDelaySrc;
      getTimeDelayList(steerFocusDelaySrc);
		std::vector<float> srcTimeStamp;
      getTimeStampList(srcTimeStamp);
		std::vector<uint> srcPulseLength;
      getPulseLengthList(srcPulseLength);

		huygen->calcFieldResponse(
			observationSpace->getResMem(),
			observationSpace->numelObsPoints(), observationSpace->getObsPoints(),
			(uint)sourceList.size(), coordSrc.data(), 
			fSrc.data(), apodSrc.data(), steerFocusDelaySrc.data(), srcTimeStamp.data(), srcPulseLength.data(),
			this->currentTime, this->currentTime, 
			observationSpace->getSpeedOfSound(), observationSpace->resultIsOnGPU());

		cpuTimeAfterCalculation = clock();
	}
}

void HuygensSimulator::updateSourceFrequencies(const float& f) {

	size_t s = sourceList.size();
	std::vector<float> fVector;
	fVector.push_back(f);

	for (size_t i = 0; i < s; ++i) {
		sourceList[i]->setFreq(fVector);
	}
}