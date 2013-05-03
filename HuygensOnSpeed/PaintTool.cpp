#include "PaintTool.h"

PaintTool::PaintTool(void)
{
}

PaintTool::~PaintTool(void)
{
}

ISource<float>* PaintTool::makePointSource(Coordinate<float> coord, float timeDelay, float timeStamp)
{
	return new PointSource(coord, frequency, uniformApod, timeDelay, pulseLength, timeStamp);
}

uint PaintTool::makeAndAddSourcesAlongLine(std::vector<ISource<float>*>* srcList, // TODO: Send source manager object.
														Coordinate<float> endPoint,
														float timeDelay, 
														float timeStmp,
														float speedOfSound)
{
	return makeAndAddSourcesAlongLine(srcList, endPoint, timeDelay, timeStmp, speedOfSound, currentApod);
}

uint PaintTool::makeAndAddSourcesAlongLine(
								std::vector<ISource<float>*>* srcList, 
								Coordinate<float> endPoint,
								float timeDelay, 
								float timeStmp,
								float speedOfSound,
								Apodization apod
								)
{
	uint n = 0;

	if (lambdaFrac > 0.05f) 
	{ 
		float lambda = speedOfSound / frequency;

		Coordinate<float> dir = Coordinate<float>::subtract(endPoint, prevPoint);
		float length = dir.length();

		dir.normalize();

		float inc = lambda * lambdaFrac; // increment
		n = (uint)ceilf(length / inc);

		if (n > 0) {

			std::vector<float> apodization = getApodVec(n, apod);

			for (uint i = 0; i < n; i++) {
				Coordinate<float> point(prevPoint.x+dir.x*inc*i, prevPoint.y+dir.y*inc*i, prevPoint.z+dir.z*inc*i);
				srcList->push_back(new PointSource(point, frequency, apodization[i], timeDelay, pulseLength, timeStmp));
			}

			printf("[%.0f ms] Created a %.1f cm, %d element ULA with %.1f*lambda spacing\n", timeStmp*1e6f, length*100, n, lambdaFrac);
		}
	} else {
		printf("Warning: lambda spacing is too small (%.1f) \n", lambdaFrac);
	}

	return n;
}

/** 
*
* Not Finished!!!
*
* Adds directive elements between startPoint and endPoint with lambda*spaceFrac spacing.
* The method returns center Coordinates of the last elem added.
*
* hight = number of points in elevation
* width = number of points in azimuth
* kerf = element spacing
* arrayNormal = array surface normal (not used, method uses start - endPoint as array cross-surface direction)
**/
Coordinate<float> PaintTool::makeAndAddDirectiveElemsAlongLine(
	std::vector<ISource<float>*>* &sourceList, 
	Coordinate<float> startPoint, 
	Coordinate<float> endPoint, 
	float freq, float spaceFrac, float speedOfSound, 
	float apod,
	float timeDelay, uint pulseLength, float timeStmp,
	uint hight, uint width, float kerf, Coordinate<float> arrayNormal)
{
	if (spaceFrac > 0.05f) 
	{ 
		float lambda = speedOfSound / freq;

		Coordinate<float> dir = Coordinate<float>::subtract(endPoint, startPoint);
		float length = dir.length();

		dir.normalize();

		float inc = lambda * spaceFrac; // increment
		uint n = (uint)ceilf(length / inc);

		if (n > 0) {

			for (uint i = 0; i < n; i++) {

				// TODO: Add source points in azimuth and elevation inside lambdaSpacing + kerf 

				Coordinate<float> point(startPoint.x+dir.x*inc*i, startPoint.y+dir.y*inc*i, startPoint.z+dir.z*inc*i);
				sourceList->push_back(new PointSource(point, freq, apod, timeDelay, pulseLength, timeStmp));
			}

			printf("[%.0f ms] Created a %.1f cm, %d (%dx%d) element ULA with %.1f*lambda spacing\n", timeStmp*1e6f, length*100, n, width, hight, spaceFrac);
		}
	} else {
		printf("Warning: lambda spacing is too small (%.1f) \n", spaceFrac);
	}

	if (sourceList->size() > 0) {
      //return sourceList->at(sourceList->size() - 1)->coord;
	} else {
		return Coordinate<float>(0, 0, 0);
	}
}

std::vector<float> PaintTool::getApodVec(int n, Apodization apod)
{
	switch (apod)
	{
	case Hann:
		return hannWindow(n);

	case Hamming:
		return hammingWindow(n);

	case Cosine:
		return cosineWindow(n);

	case Uniform: 
	default:
		return uniformWindow(n);
	}
}

std::vector<float> PaintTool::hannWindow(int n)
{
	std::vector<float> window = std::vector<float>(n);

	for (int i = 0; i < n; i++)
	{
		window[i] = 0.5f*(1 - cos(2.0f*PI*i/(n-1)));
	}

	return window;
}

std::vector<float> PaintTool::hammingWindow(int n)
{
	std::vector<float> window = std::vector<float>(n);
	
	for (int i = 0; i < n; i++)
	{
		window[i] = 0.54f - 0.46f*cos(2.0f*PI*i/(n-1));
	}

	return window;
}

std::vector<float> PaintTool::cosineWindow(int n)
{
	std::vector<float> window = std::vector<float>(n);
	
	for (int i = 0; i < n; i++)
	{
		window[i] = sin(PI*i/(n-1));
	}

	return window;
}

std::vector<float> PaintTool::uniformWindow(int n)
{
	std::vector<float> window = std::vector<float>(n);
	
	for (int i = 0; i < n; i++)
	{
		window[i] = uniformApod;
	}

	return window;
}

Apodization PaintTool::nextApod()
{
	currentApod = (Apodization) ( (currentApod + 1) % NUMBER_OF_APOD_FUNC ); 
	return currentApod;
}