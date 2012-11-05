/**
* GLUT user interface to the Huygens Principle Simulator
**/

#include "HuygensOnSpeed.h"

void makeNewObservationSpace() // move to the simulator class
{
	// TODO: make a copy constructor in ObservationSpace
	Coordinate<float> minLim = observationSpace->getMinLim();
	Coordinate<float> maxLim = observationSpace->getMaxLim();
	float sos = observationSpace->getSpeedOfSound();
	bool resultOnGPU = observationSpace->resultIsOnGPU();

	delete observationSpace;

	observationSpace = new ObservationArea(dim, minLim, maxLim, samplesPerMeter, sos, resultOnGPU);
}

// GLUT idle callback function
void idle() { 
	glutPostRedisplay();
}

void printHelpMenu()
{
	printf("\n----------------------------HELP MENU----------------------------\n");
	printf(
		"\t Left click: \t\t new source\n"
		"\t Left click & drag: \t multiple sources\n"
		"\t Right click & drag: \t steer and focus\n"
		"\t Resize window: \t enlarge obs area\n"
		"\t O/o: \t\t\t +- obs samples per mm\n"
		"\t F/f: \t\t\t +- frequency\n"
		"\t T/t: \t\t\t +- simulation time\n"
		"\t S/s: \t\t\t +- simulation speed\n"
		"\t D/d: \t\t\t +- dynamic range\n"
		"\t L/l: \t\t\t +- lambda spacing\n"
		"\t N/n: \t\t\t +- pulse length\n"
		"\t a: \t\t\t select apodization function\n"
		"\t m: \t\t\t cw or pw mode\n"
		"\t .: \t\t\t line or free hand draw mode\n"
		"\t x: \t\t\t list sources\n"
		"\t c: \t\t\t clear source list\n"
		"\t e: \t\t\t display envelope or phase\n"
		"\t z: \t\t\t zoom on and off\n"
		"\t r: \t\t\t reset timer\n"
		"\t p: \t\t\t pause\n"
		"\t h and unused keys: \t this menu\n"
		"\t q: \t\t\t quit\n"
		"\n"
		" hint: start app with -v for vascular setup (cardiac is default)\n");
	printf("----------------------------------------------------------------\n");
	printf("2011 - Jon Petter AAsen - jon.p.asen@ntnu.no\n");
	printf("----------------------------------------------------------------\n");
}

void updateWindowTitle()
{
	//char* title = new char[256];
	snprintf(title, 512, 
		"%s [%s] |%.2f us| |%.0f dB| |%d #src| |%.1f MHz %.1f mm| |%s| |%dK #obs| |%.1fx%.1f mm| |%.1f #/mm| |%5.1f fps|", 
		applicationTitle, 
		simulator->isInPlay()? "|>" : "||",
		simulator->getCurrentTime()*1000000,
		dispResp->getDynamicRange(),
		simulator->numberOfPointSources(),
		paintTool->getFreq()/1000000,
		(observationSpace->getSpeedOfSound()/paintTool->getFreq())*1000,
		cwMode? "CW" : "PW",
		observationSpace->numelObsPoints()/1000,
		observationSpace->areaSizeX()*1000,
		observationSpace->areaSizeZ()*1000,
		samplesPerMeter/1000.0f,
		simulator->getFrameRate());

	glutSetWindowTitle(title);
	//delete [] title;
}

// GLUT keybord callback function
void keyboard(unsigned char key, int x, int y)
{
	switch(key) 
	{
	case 'q': cleanUpAtExit(); exit(0); break; // TODO: memory error when hitting q when cleanUpAtExcit is called.

	case 'f': paintTool->decreaseFrequency(); simulator->updateSourceFrequencies(paintTool->getFreq()); break;
	case 'F': paintTool->increaseFrequency(); simulator->updateSourceFrequencies(paintTool->getFreq()); break;

	case 'T': simulator->stepForward(); break;
	case 't': simulator->stepBack(); break;

	case 'S': 
		simulator->increaseSimSpeed(); 
		printf("Sim s in CPU s: %.2f\n", simulator->getSimCPUTimeFactor()); break;
	case 's': 
		simulator->decreaseSimSpeed(); 
		printf("Sim s in CPU s: %.2f\n", simulator->getSimCPUTimeFactor()); break;

	case 'L': 
		paintTool->increaseLambdaFrac();
		printf("Lambda frac: %.1f\n", paintTool->getLambdaFrac()); 
		break;
	case 'l': 
		paintTool->decreaseLambdaFrac();
		printf("Lambda frac: %.1f\n", paintTool->getLambdaFrac()); 
		break;

	case 'd': dispResp->updateDynamicRange(-5.0f); break; 
	case 'D': dispResp->updateDynamicRange(5.0f); break;

	case 'o':
		samplesPerMeter -= 100;
		if (samplesPerMeter < 100) samplesPerMeter = 100;
		makeNewObservationSpace();
		break;
	case 'O':
		samplesPerMeter += 100;
		makeNewObservationSpace();
		break;

	case 'n': 
		if (!cwMode) { 
			if (paintTool->getPulseLength() > 1) {
				paintTool->decreasePulseLength();
			}
		}
		printf("Pulse length: %.1f\n", paintTool->getPulseLength());
		break;
	case 'N': 
		if (!cwMode) {
			paintTool->increasePulseLength();
		}
		printf("Pulse length: %.1f\n", paintTool->getPulseLength());
		break;

	case 'x': simulator->listSources(); break;

	case 'z': 
		zoomMode = !zoomMode; 
		printf(zoomMode? "Resize: zoom mode\n" : "Resize: extend obs mode\n");
		break;

	case 'r': simulator->restart();  break;
	case 'c': simulator->clearSrcList(); break;
	case 'e': 
		dispResp->switchDispMode(); 
		printf(dispResp->dispEnvelopeIsOn()? "Display envelope\n" : "Display phase\n"); 
		break;

	case 'a': 
		printf("Apod = %d (0 = uniform, 1 = hann, 2 = hamming, 3 = cosine)\n", paintTool->nextApod() );
		break;

	case 'm': cwMode = !cwMode; break; 

	case '.': freeDrawMode = !freeDrawMode; printf(freeDrawMode? "Array draw mode: free hand\n" : "Array draw mode: linear\n"); break;

	case 'p': simulator->playPause(); break;

	case 'h': 
	default: printHelpMenu(); break;
	}

	// set correct pulse length depending on cw or pw mode
	if (key == 'm' || key == 'n' || key == 'N') {
		if (key == 'm') {
			if (cwMode)	{	
				paintTool->setPulseLength(0.0f);
			} else {	
				paintTool->setPulseLength(10.0f);
			}
		}
	}

	glutPostRedisplay();
}

Coordinate<float> mapScreenCoordsToSimulatorCoords(ObservationArea *oa, 
												   IDisplayResponse *dr, 
												   int screenX, int screenY)
{
	return oa->getPosition(screenX, dr->getHeight()-screenY, dr->getWidth(), dr->getHeight());
}

// GLUT mouse clicked callback function
void mouse(int button, int state, int x, int y)
{
	if (button == GLUT_LEFT_BUTTON) // source mode
	{
		focusMode = false;

		int z = y;

		if (x > 0 && z > 0 && x < dispResp->getWidth() && z < dispResp->getHeight()) 
		{
			Coordinate<float> c = mapScreenCoordsToSimulatorCoords(observationSpace, dispResp, x, z); 

			if (state == GLUT_UP) 
			{
				if (mouseHasBeenMoved)
				{
					if (!freeDrawMode) {
						uint n = paintTool->makeAndAddSourcesAlongLine(
							simulator->getSourceList(), 
							c, 
							0.0f, 
							simulator->getCurrentTime(), 
							observationSpace->getSpeedOfSound());
						simulator->updateNumberOfPointSources(n);
					}
				} 
				else 
				{
					//sourceList.push_back(new Source(c, currentFreq, 1.0f, 0.0f, pulseLength, currentTime));
					simulator->addSrc(paintTool->makePointSource(c, 0.0f, simulator->getCurrentTime())); // TODO: get current time from simulator
					printf("[%.0f us] Added source point at (%.2f, %.2f, %.2f)\n", simulator->getCurrentTime()*1000000, c.x, c.y, c.z);
				}

				mouseButtonIsDown = false;
				mouseHasBeenMoved = false;
			} 
			else if (state == GLUT_DOWN) // register initial mouse click
			{
				mouseButtonIsDown = true;
				paintTool->setPrevCoord(c);
				paintTool->setPrevCoordFreeMode(c);
				//prevPoint = c;
				//prevPointFreeMode = c;

				prevMousePos.x = x;
				prevMousePos.y = y;
			}
		}
	}
	else if (button == GLUT_RIGHT_BUTTON) // focus mode
	{
		if (state == GLUT_DOWN) { // enter focus mode
			focusMode = true;
		} else if (state == GLUT_UP) { // exit focus mode
			focusMode = false;
		}
	}
	glutPostRedisplay();
}

// GLUT mouse moved callback function
void motion(int x, int y) 
{
	if (mouseButtonIsDown) 
	{
		if (prevMousePos.x != x || prevMousePos.y != y)
		{
			mouseHasBeenMoved = true;
		}

		if (freeDrawMode) 
		{
			int z = y;
			Coordinate<float> c = mapScreenCoordsToSimulatorCoords(observationSpace, dispResp, x, z); 

			float lengthSinceLastPoint = Coordinate<float>::subtract(c, paintTool->getPrevCoordFreeMode()).length();

			if (lengthSinceLastPoint > (paintTool->getLambdaFrac() * (observationSpace->getSpeedOfSound() / paintTool->getFreq()))) {
				//sourceList.push_back(new Source(c, currentFreq, 1.0f, 0.0f, pulseLength, currentTime));
				simulator->addSrc(paintTool->makePointSource(c, 0.0f, simulator->getCurrentTime()));
				paintTool->setPrevCoordFreeMode(c);
				//prevPointFreeMode = c;
				
				// free hand lambda-spaced drawing. Will only work in pause mode 
				//Coordinate<float> lastPointAdded = PaintTool::makeAndAddSourcesAlongLine(sourceList, prevPointFreeMode, c, currentFreq, lambdaFrac, speedOfSound, 1.0f, 0.0f, pulseLength, currentTime);
				//prevPointFreeMode = lastPointAdded;
			}
		}
	} 
	else 
	{
		if (focusMode) 
		{
			int z = y;
			Coordinate<float> c = mapScreenCoordsToSimulatorCoords(observationSpace, dispResp, x, z); 
			simulator->updateTimeDelays(c);
		}
	}
	glutPostRedisplay();
}

// GLUT arrow keys callback function 
void arrow_keys( int a_keys, int x, int y ) {}

// GLUT window reshape callback function
void reshape(int x, int y) {

	if (x > 0 && y > 0) 
	{
		// update display object
		dispResp->setDispDim(Dimension<int>(x,y));

		glViewport(0, 0, x, y);
		glutReshapeWindow(x,y);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0); 

		if (!zoomMode) 
		{
			// make a new observation area.
			Coordinate<float> maxLim( meterPerPixel*dispResp->getWidth()/2, 0.00f,  meterPerPixel*dispResp->getHeight()/2);
			Coordinate<float> minLim(-meterPerPixel*dispResp->getWidth()/2, 0.00f, -meterPerPixel*dispResp->getHeight()/2);	
			float sos = observationSpace->getSpeedOfSound();
			bool resultOnGPU = observationSpace->resultIsOnGPU();

			if (observationSpace) delete observationSpace;
			observationSpace = new ObservationArea(dim, minLim, maxLim, samplesPerMeter, sos, resultOnGPU); // TODO: make a copy constructor in ObservationArea
		}
	}
}

void display()
{

#ifdef ADD_PROFILER_ARRAY // Add debug array
	if (++numberOfArraysAdded <= numberOfArrays) 
	{
		paintTool->setPrevCoord(mapScreenCoordsToSimulatorCoords(observationSpace, dispResp, 100, 75));
		Coordinate<float> c = mapScreenCoordsToSimulatorCoords(observationSpace, dispResp, 100, 175); 
		uint n = paintTool->makeAndAddSourcesAlongLine(
			simulator->getSourceList(), 
			c, 
			0.0f, 
			-0.001f, 
			observationSpace->getSpeedOfSound());
		simulator->updateNumberOfPointSources(n);
	}
#endif

	//clock_t time1 = clock();
	//clock_t time2;

	// calculate field respons
	simulator->calcSimulation(observationSpace);

	//time2 = clock();
	//float t1 = 1000.0f*(time2 - time1)/float(CLOCKS_PER_SEC);
	//printf("Simulation %.2f ms\n", t1));

	// map result to display buffer
	dispResp->mapResponseToDisplay(
		observationSpace->getResMem(), 
		observationSpace->nObsPoints().x, 
		observationSpace->nObsPoints().z,
		observationSpace->resultIsOnGPU());

	//time1 = clock();
	//float t2 = 1000.0f*(time1 - time2)/float(CLOCKS_PER_SEC);
	//printf("Display %.2f ms\n", t2);

	updateWindowTitle();

	glutSwapBuffers(); // strange lag when swapping buffers. TODO: Change display-class to use textures!

	//time2 = clock();
	//float t3 = 1000.0f*(time2 - time1)/float(CLOCKS_PER_SEC);
	//printf("Update window %.2f ms\n", t3);

	//if (t1 < t3)
	//	printf("Display %.2f ms\n", t3);
}

// GLUT atExit clean up function
void cleanUpAtExit() 
{
	if (observationSpace) delete observationSpace;
	//if (huygen) delete huygen; // is deleted by the simulator
	if (dispResp) delete dispResp;
	if (paintTool) delete paintTool;
	if (simulator) delete simulator;

	observationSpace = NULL;
	dispResp = NULL;
	paintTool = NULL;
	simulator = NULL;

#ifdef _DEBUG
	_CrtDumpMemoryLeaks();
#endif
}

void initApplication(int argc, char **argv)
{

	if (argc > 1) {
		if (strcmp("-v", argv[1]) == 0) {
			vascular = true;
			cardiac  = false;
		} 
	}
	if (vascular) {
		dispResp = new DisplayResponse(Dimension<int>(640,640));
	} else { // standard -> cardiac
		dispResp = new DisplayResponse(Dimension<int>(800,250));//640,320));
	}

#ifdef CALC_ON_GPU
	huygen = new HuygensOnGPU();
#else
	huygen = new HuygensOnCPU();
#endif

	simulator = new HuygensSimulator(huygen);

	if (vascular) {
		// Vascular setup
		// f0 = 7.5MHz, spacing = lambda, pulseLength = 0 -> Inf
		paintTool = new PaintTool(7.5e+6f, 1.0f, 0.0f);
	} else {
		// Cardiac setup
		// f0 = 2.5MHz, spacing = lambda/2, pulseLength = 0 => Inf
		paintTool = new PaintTool(2.5e+6f, 0.5f, 0.0f);
	}
}

void initObservationSpace()
{
	// observation space related
	dim = 2;

	if (vascular) {
		// vascular setup
		meterPerPixel = 6e-5f;
	} else {
		// cardiac setup
		meterPerPixel = 2e-4f; // so one pixel is always 0.2 mm in observation space.
	}
	Coordinate<float> maxLim( meterPerPixel*dispResp->getWidth()/2, 0.00f,  meterPerPixel*dispResp->getHeight()/2);
	Coordinate<float> minLim(-meterPerPixel*dispResp->getWidth()/2, 0.00f, -meterPerPixel*dispResp->getHeight()/2);	
	
	if (vascular) {
		// vascular setup
		samplesPerMeter = 12000;
	} else {
		// cardiac setup
		samplesPerMeter = 4000; // Test setup!!! 4000
	}

	float speedOfSound = 1540;		// m/s
	bool resultOnGPU = true;

	observationSpace = new ObservationArea(dim, minLim, maxLim, samplesPerMeter, speedOfSound, resultOnGPU);
}

void initCUDA(int argc, char **argv) 
{
	int dev = 1;
	cuUtilsSafeCall( cudaGetDeviceCount(&dev) );
	cuUtilsSafeCall( cudaSetDevice(dev-1) );
	//cuUtilsSafeCall( cudaDeviceReset() );
	cuUtilsSafeCall( cudaGLSetGLDevice(dev-1) ); // pick the first GPU to link with openGL context of this thread
}

void initGLUT(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);

	if (argc > 2) {
		dispResp->setDispDim(Dimension<int>(atoi(argv[1]), atoi(argv[2])));
	}

	glutInitWindowSize(dispResp->getWidth(), dispResp->getHeight());
	glutInitWindowPosition(
		(glutGet(GLUT_SCREEN_WIDTH) - dispResp->getWidth())/2, 
		(glutGet(GLUT_SCREEN_HEIGHT) - dispResp->getHeight())/2);

    glutCreateWindow(applicationTitle);

	// set glut callback functions
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutReshapeFunc(reshape);
	glutIdleFunc(idle);
	//glutSpecialFunc(arrow_keys);
}

void initGL(int argc, char **argv)
{	
}

int main(int argc, char** argv) 
{
	initApplication( argc, argv );

	//initGL( argc, argv );

	initGLUT( argc, argv );

	GLenum err = glewInit();

	if (err || !glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object")) 
	{
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(-1);

	} else {

		initCUDA( argc, argv );

		initObservationSpace();

		// init the pbo
		dispResp->createPBO(); // Create pixel buffer object

		// init result device memory
		//createResMem();

		atexit(cleanUpAtExit);

		//
		printHelpMenu();

		// start glut main loop
		glutMainLoop();

	}
	return 0;
}