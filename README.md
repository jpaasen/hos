# HOS - Huygens On Speed

GPU-based ultrasound field simulator with interactive user interface.

Homepage: http://heim.ifi.uio.no/jpaasen/huygens/

Released under the GPL license.

## What do you need to make it run on speed?
- [CUDA capable GPU](https://developer.nvidia.com/cuda-gpus)
- [CUDA Toolkit 3.2 or greater](https://developer.nvidia.com/cuda-downloads)

## Build with cmake and make
1. Create "build" folder
2. Enter build folder and type: `>> cmake ..`
3. Then make the project with: `>> make`
4. Run program with: `>> ./HuygensOnSpeedGlutApp/HuygensOnSpeedApp`

Or short: `>> cmake .. ; make ; ./HuygensOnSpeedGlutApp/HuygensOnSpeedApp`

## Build with cmake and Visual Studio
1. Generate Visual Studio 20XY solution file using cmake in terminal or GUI.
2. Open solution file and build (F5).

## Build options
### Build CPU version
Set CMAKE arguments `CALC_WITH` and `DISP_WITH` to either `"CUDA"` or `"CPU"`.
Valid combinations are:
 - `CALC_WITH="CUDA"` and `DISP_WITH="CUDA"`
 - `CALC_WITH="CPU"` and `DISP_WITH="CUDA"`
 - `CALC_WITH="CPU"` and `DISP_WITH="CPU"`
Command line example:  
`>> cmake .. -DCALC_WITH="CPU" -DDISP_WITH="CPU"`

## Info
### Info for Linux users
Remember to have libGL, libGlew, libGLU and libglut in lib/linker path.
To grab glew use: sudo apt-get install libglew1.6-dev
To grab glu use: sudo apt-get install libglu1-mesa-dev
The other two is usually install together with CUDA.
To se if you have a lib installed use: apt-file search <libGLU> 

### Info for Windows users
Generate solution file using cmake.
Project dependencies (libs) are found in the CUDA Toolkit and SDK if `CALC_WITH="CUDA"` or `DISP_WITH="CPU"` and should be located automatically by cmake. If both `CALC_WITH` and `DISP_WITH` are `"CPU"`, the project uses files found in the local include and lib folder.

### Info for Mac users
Same build instructions as for linux: `mkdir build; cd build; cmake ..; make`

Glew have to be install to make it run.
`>> brew install glew`
If you don't have brew you can get it [here](http://brew.sh)
Or just paste the following line in the terminal: `ruby -e "$(curl -fsSL https://raw.github.com/Homebrew/homebrew/go/install)"`

#### Known issues on mac
Have to force close the application (`ctrl + z` in the terminal and right-klick + force quit on the ui icon)

## Info about the projects

### HuygensOnSpeed
This includes the simulator code and data structures for point sources, observation points, Paint tools and etc. It includes both a GPU and CPU implementation. For display, a CUDA-OpenGL binding is used. Now there is also a display class for CPU and OpenGL only.

#### Build instructions (windows)
The following build instructions is based on this code beeing build as a static library HuygensOnSpeed(.lib/.a).
* Additional include directories: `"$(CUDA_PATH)\include";"$(NVSDKCOMPUTE_ROOT)\C\common\inc"`
* Additional lib dependencies: `cudart(.lib) cublas(.lib)`
* Additional lib dirs: `"$(NVSDKCOMPUTE_ROOT)/C/common/lib/$(PlatformName)"; "$(CUDA_PATH)/lib/$(PlatformName)"`

## HuygensOnSpeedGlutApp
This is the Paint-UI used to draw lines of source points.

### Build instructions (windows)
* Additional include directories: `"$(CUDA_PATH)\include";"$(NVSDKCOMPUTE_ROOT)\C\common\inc"`
* Additional lib dependencies: `HuygensOnSpeed(.lib) glew(64.lib)`
* Additional lib dirs: `"$(NVSDKCOMPUTE_ROOT)/C/common/lib/$(PlatformName)";"$(SolutionDir)$(PlatformName)/$(ConfigurationName)";"$(CUDA_PATH)/lib/$(PlatformName)"`

After compilation (on windows, maybe on Linux too?), the glut app will require the following dynamic libraries from the cuda sdk in order to run: 
* cudart(64.dll)
* cublas(64.dll) 
* freeglut(.dll)
* glew(64.dll)

## MatlabInterface
Functionality for calling HuygensOnSpeed from Matlab.

### Build instructions
See own Readme file in sub folder. Building mex-interface has only been tested on windows 7 64bit.

Have fun!
Jon Petter Åsen - jon.p.asen@ntnu.no

