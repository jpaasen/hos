/**
* Part of the ...
*
* Huygen's Principle Simulator - Huygen on Speed
*
* Author: Jon Petter Åsen
* E-mail: jon.p.asen@ntnu.no - jp.aasen@gmail.com
**/
#pragma once

#include "Coordinate.h"
#include "ISource.h"
#include "PointSource.h"
#include "HuygensSimulator.h"

#include <vector>
#include <math.h>

#define NUMBER_OF_APOD_FUNC 4

enum Apodization
{
	Uniform,
	Hann,
	Hamming,
	Cosine
};

class PaintTool
{
private:
	Coordinate<float> prevPoint;
	Coordinate<float> prevPointFreeMode;

	float lambdaFrac;
	float frequency;
	float pulseLength;

	float uniformApod;
	Apodization currentApod;
	//int currentApod;
	//Apodization apodList[] = {Uniform, Hann, Hamming, Cosine};

	float deltaLambdaFrac;
	float deltaFrequency;
	float deltaPulseLength;

public:
	PaintTool(void);

	PaintTool(float initFreq, float initLambdaFrac, float initPulseLength)
	{
		frequency	= initFreq;
		lambdaFrac	= initLambdaFrac;
		pulseLength = initPulseLength;
		uniformApod = 1.0f;

		deltaLambdaFrac		= 0.1f;
		deltaFrequency		= 100000.0f;
		deltaPulseLength	= 1.0f;

		prevPoint			= Coordinate<float>::zeros();
		prevPointFreeMode	= Coordinate<float>::zeros();	

		currentApod = Uniform;
	}

	~PaintTool(void);

	ISource<float>* makePointSource(Coordinate<float> coord, float timeDelay, float timeStamp);

	/** 
	* Adds src points between this.prevPoint and endPoint with this.lambda*this.spaceFrac spacing.
	* Uniform apodization is used.
	* The method returns how many point sources where added.
	**/
	unsigned int makeAndAddSourcesAlongLine(
		std::vector<ISource<float>*>* srcList, 
		Coordinate<float> endPoint,
		float timeDelay, 
		float timeStmp,
		float speedOfSound
		);

	/** 
	* Adds src points between this.prevPoint and endPoint with this.lambda*this.spaceFrac spacing.
	* Apodized by the given vector. Length 1 or equal length of 
	* The method returns how many point sources where added.
	**/
	unsigned int makeAndAddSourcesAlongLine(
		std::vector<ISource<float>*>* srcList, 
		Coordinate<float> endPoint,
		float timeDelay, 
		float timeStmp,
		float speedOfSound,
		Apodization apod
		);

	static Coordinate<float> makeAndAddDirectiveElemsAlongLine(
																std::vector<ISource<float>*>* &sourceList, 
																Coordinate<float> startPoint, 
																Coordinate<float> endPoint, 
																float freq, 
																float spaceFrac, 
																float speedOfSound, 
																float apod,
																float timeDelay, 
																unsigned int pulseLength, 
																float timeStmp,
																unsigned int hight, 
																unsigned int width, 
																float kerf, 
																Coordinate<float> arrayNormal
																);

	std::vector<float> getApodVec(int n, Apodization apod);
	
	std::vector<float> hannWindow(int n);

	std::vector<float> hammingWindow(int n);

	std::vector<float> cosineWindow(int n);

	std::vector<float> uniformWindow(int n);

	// update function
	void decreaseLambdaFrac() 
	{
		if (lambdaFrac > deltaLambdaFrac)
		{
			lambdaFrac -= deltaLambdaFrac;
		}
	}
	void increaseLambdaFrac() {lambdaFrac += deltaLambdaFrac;}

	void decreaseFrequency() 
	{
		if (frequency > deltaFrequency)
		{
			frequency -= deltaFrequency;
		}
	}
	void increaseFrequency() {frequency += deltaFrequency;}

	void decreasePulseLength() 
	{
		if (pulseLength > deltaPulseLength)
		{
			pulseLength -= deltaPulseLength;
		}
	}
	void increasePulseLength() {pulseLength += deltaPulseLength;}

	// geters and seters
	Coordinate<float> getPrevCoord() {return prevPoint;}
	void setPrevCoord(Coordinate<float> coord) {prevPoint = coord;}

	Coordinate<float> getPrevCoordFreeMode() {return prevPointFreeMode;}
	void setPrevCoordFreeMode(Coordinate<float> coord) {prevPointFreeMode = coord;}

	float getLambdaFrac() {return lambdaFrac;}
	void setLambdaFrac(float lambdaFrac) {this->lambdaFrac = lambdaFrac;}

	float& getFreq() {return frequency;}
	void setFrequency(float freq) {frequency = freq;}

	float getPulseLength() {return pulseLength;}
	void setPulseLength(float pulseLength) {this->pulseLength = pulseLength;}

	float getUniformApod() {return uniformApod;}
	void setUniformApod(float apod) {uniformApod = apod;}


	float getDeltaLambdaFrac() {return deltaLambdaFrac;}
	void setDeltaLambdaFrac(float dLambdaFrac) {deltaLambdaFrac = dLambdaFrac;}

	float getDeltaFrequency() {return deltaFrequency;}
	void setDeltaFrequency(float deltaFreq) {deltaFrequency = deltaFreq;}

	float getDeltaPulseLength() {return deltaPulseLength;}
	void setDeltaPulseLength(float dPulseLength) {deltaPulseLength = dPulseLength;}

	Apodization nextApod();

};
