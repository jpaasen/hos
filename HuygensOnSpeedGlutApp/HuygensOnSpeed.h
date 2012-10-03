/**
* Part of the ...
*
* Huygen's Principle Simulator - Huygen on Speed
*
* Author: Jon Petter Åsen
* E-mail: jon.p.asen@ntnu.no - jp.aasen@gmail.com
**/
#pragma once

// We use the GPU version of the Huygens principle calculator.
#define CALC_ON_GPU
//#define CALC_ON_CPU

#ifdef _WIN32
#  include <windows.h>
#endif

#include <GL/glew.h>
// Include GLUT header
#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
//#include <GL/glut.h>
#include <GL/freeglut.h>
#endif

#include "../HuygensOnSpeed/Defines.h"

#include "GlutCallbackFunctions.h"

#include "../HuygensOnSpeed/HuygensOnGPU.h"
#include "../HuygensOnSpeed/HuygensOnCPU.h"
#include "../HuygensOnSpeed/DisplayResponse.h"
#include "../HuygensOnSpeed/PaintTool.h"
#include "../HuygensOnSpeed/HuygensSimulator.h"
#include "../HuygensOnSpeed/Coordinate.h"
#include "../HuygensOnSpeed/Dimension.h"
#include "../HuygensOnSpeed/Source.h"
#include "../HuygensOnSpeed/ObservationArea.h"

// std libs
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <time.h>
#include <math.h>
//#include <algorithm>
//#include <map>
//#include <string>

//Code for detecting memory leaks
#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

#define DEBUG_NEW new(_NORMAL_BLOCK, __FILE__, __LINE__)
#define new DEBUG_NEW
#endif

//

char *applicationTitle = "Huygens On Speed"; // Title string for this application
char title[512];

IDisplayResponse* dispResp;		// display interface
HuygensSimulator* simulator;	// simulator interface
PaintTool* paintTool;			// paint tool for adding point sources and array sources

IHuygensPrinciple<float, cuComplex>* huygen; // move to the simulator class

float meterPerPixel; // display? / simulator?

int dim; // move to observation space class
float samplesPerMeter; // move to observation space class
ObservationArea* observationSpace; // move to the simulator class

// drawing parameters. GUI-parameters. Move to header. Make a GUI class.
bool mouseButtonIsDown = false;
bool mouseHasBeenMoved = false;
bool focusMode = false;
bool cwMode = true;
bool freeDrawMode = false;
bool zoomMode = false;

bool cardiac  = true; // use cardiac setup
bool vascular = false;  // use vascular setup

Coordinate<int> prevMousePos(0,0,0);  // display or UI

// forward declerations

//void createPBO();

void updateWindowTitle();

//#define ADD_PROFILER_ARRAY
#ifdef ADD_PROFILER_ARRAY
int numberOfArrays = 20;
int numberOfArraysAdded = 0;
#endif