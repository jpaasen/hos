Welcom to the HuygensOnSpeed simulator

####################################
# What do you need to make it run? #
####################################
- CUDA capable GPU
- CUDA 3.2 or greater

###############
# CMAKE BUILD #
###############

1. Create "build" folder
2. Enter build folder and type:
   >> cmake ..
3. Then make the project with:
   >> make
4. Run program with:
   >> ./HuygensOnSpeedGlutApp/HuygensOnSpeedApp

Or short: cmake .. ; make ; ./HuygensOnSpeedGlutApp/HuygensOnSpeedApp


######################
# Visual Studio 2008 #
######################

1. Open HuygensOnSpeed.sln
2. Select Win32/x64 and Release/Debug
3. Build and run (F5)
(Solution file can also be generated using cmake)


#############
# Projects: #
#############

*** HuygensOnSpeed ***
- This includes the simulator code and data structures for point sources, observation points, Paint tools and etc. 
  It includes both a GPU and CPU implementation. For display, a CUDA-OpenGL binding is used.

 Build instructions: (windows)
  The following build instructions is based on this code beeing build as a static library HuygensOnSpeed(.lib/.a).
  Additional include directories: "$(CUDA_PATH)\include";"$(NVSDKCOMPUTE_ROOT)\C\common\inc"
  Additional lib dependencies: cudart(.lib) cublas(.lib)
  Additional lib dirs: "$(NVSDKCOMPUTE_ROOT)/C/common/lib/$(PlatformName)"; "$(CUDA_PATH)/lib/$(PlatformName)"

*** HuygensOnSpeedGlutApp ***
- This is the Paint-UI used to draw lines of source points.

 Build instructions: (windows)
  Additional include directories: "$(CUDA_PATH)\include";"$(NVSDKCOMPUTE_ROOT)\C\common\inc"
  Additional lib dependencies: HuygensOnSpeed(.lib) glew(64.lib)
  Additional lib dirs: "$(NVSDKCOMPUTE_ROOT)/C/common/lib/$(PlatformName)";"$(SolutionDir)$(PlatformName)/$(ConfigurationName)";"$(CUDA_PATH)/lib/$(PlatformName)"
  After compilation (on windows, maybe on Linux too?), the glut app will require the following dynamic libraries from the cuda sdk in order to run: 
   - cudart(64.dll)
   - cublas(64.dll) 
   - freeglut(.dll)
   - glew(64.dll)

*** MatlabInterface ***
- Functionality for calling HuygensOnSpeed from Matlab.

 Build instructions
  See own Readme file in sub folder.

#########
# Linux #
#########
Remember to have libGL, libGlew and libglut in lib/linker path.

Best regards
Jon Petter Åsen - jon.p.asen@ntnu.no
