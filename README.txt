Welcom to the HuygensOnSpeed simulator

Projects:

HuygensOnSpeed 
- This includes the simulator code and data structures for point sources, observation points, Paint tools and etc. It includes both a GPU and CPU implementation. For display, a CUDA-OpenGL binding is used.
Build instructions
Additional include directories: "$(CUDA_PATH)\include";"$(NVSDKCOMPUTE_ROOT)\C\common\inc"
Additional lib dependencies: cudart(.lib) cublas(.lib)
Additional lib dirs: "$(NVSDKCOMPUTE_ROOT)/C/common/lib/$(PlatformName)"; "$(CUDA_PATH)/lib/$(PlatformName)"
The following build instructions is based on this code beeing build as a static library HuygensOnSpeed(.lib/.a).

HuygensOnSpeedGlutApp
- This is the Paint-UI used to draw lines of source points.
Build instructions
Additional include directories: "$(CUDA_PATH)\include";"$(NVSDKCOMPUTE_ROOT)\C\common\inc"
Additional lib dependencies: HuygensOnSpeed(.lib) glew(64.lib)
Additional lib dirs: "$(NVSDKCOMPUTE_ROOT)/C/common/lib/$(PlatformName)";"$(SolutionDir)$(PlatformName)/$(ConfigurationName)";"$(CUDA_PATH)/lib/$(PlatformName)"
After compilation, the glut app will require the following dynamic libraries from the cuda sdk in order to run: 
 -cutil(64.dll) and 
 -freeglut(.dll)

MatlabInterface
- Functionality for calling HuygensOnSpeed from Matlab.
Build instructions
See own Readme file in sub folder.

Best regards
Jon Petter Åsen - jon.p.asen@ntnu.no