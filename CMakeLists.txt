cmake_minimum_required(VERSION 2.8)

set( CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if (NOT DEFINED CALC_WITH)
	set(CALC_WITH "CUDA")
endif ()
if (NOT DEFINED DISP_WITH)
	set(DISP_WITH "CUDA")
endif ()

if (CALC_WITH STREQUAL CPU)
	add_definitions( -DCALC_WITH_CPU )
elseif (CALC_WITH STREQUAL OPENCL)
	add_definitions( -DCALC_WITH_OPENCL )
else ()
	add_definitions( -DCALC_WITH_CUDA )
endif ()

if (DISP_WITH STREQUAL CPU)
	add_definitions( -DDISP_WITH_CPU )
elseif (DISP_WITH STREQUAL OPENCL)
	add_definitions( -DDISP_WITH_OPENCL )
else ()
	add_definitions( -DDISP_WITH_CUDA )
endif ()

if (CALC_WITH_CUDA STREQUAL CUDA OR DISP_WITH STREQUAL CUDA)
   find_package(CUDA)
   include(FindCUDA)
   if ( UNIX )
      find_package(GLUT)
      find_package(GLEW)
   endif ()

   link_directories(${CUDA_LIBRARIES})
   include_directories(${CUDA_INCLUDE_DIRS})

   if ( WIN32 )
      include_directories("${CUDA_SDK_ROOT_DIR}/common/inc")
      if (CMAKE_CL_64)
         link_directories( ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64 )
         link_directories( ${CUDA_SDK_ROOT_DIR}/common/lib/x64 )
      else ()
         link_directories( ${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32 )
         link_directories( ${CUDA_SDK_ROOT_DIR}/common/lib/Win32 )
      endif ()
      set(CUDA_NVCC_FLAGS "-use_fast_math")
      #set(CUDA_NVCC_FLAGS "-G0") # for debugging gpu kernels 
   else()
      set(CUDA_NVCC_FLAGS "-use_fast_math")
      #set(CUDA_NVCC_FLAGS "-G") # for debugging gpu kernels 
   endif ()
else ()
   if ( UNIX )
      find_package(GLUT)
      find_package(GLEW)
   elseif ( WIN32 )
      if (CMAKE_CL_64)
	     link_directories("${CMAKE_CURRENT_SOURCE_DIR}/lib/x64")
      else ()
	     link_directories("${CMAKE_CURRENT_SOURCE_DIR}/lib/Win32")
	  endif ()
      include_directories("include")
   endif ()
endif ()

add_subdirectory( HuygensOnSpeed )
add_subdirectory( HuygensOnSpeedGlutApp )

# Uncomment to see cmake variables
#get_cmake_property(_variableNames VARIABLES)
#foreach (_variableName ${_variableNames})
#message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()


 
 
