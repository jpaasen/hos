cmake_minimum_required(VERSION 2.8)

set( CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(CUDA)
include(FindCUDA)
if ( UNIX )
   find_package(GLUT)
   find_package(GLEW)
endif ()

# Uncomment to see cmake variables
#get_cmake_property(_variableNames VARIABLES)
#foreach (_variableName ${_variableNames})
#message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

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
endif ()

add_subdirectory( HuygensOnSpeed )
add_subdirectory( HuygensOnSpeedGlutApp )


 
 
