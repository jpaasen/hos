cmake_minimum_required(VERSION 2.8)

set( CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(CUDA)
find_package(GLUT)
find_package(GLEW)

get_cmake_property(_variableNames VARIABLES)
foreach (_variableName ${_variableNames})
message(STATUS "${_variableName}=${${_variableName}}")
endforeach()

link_directories(${CUDA_LIBRARIES})
# message(${GLEW_LIBRARIES})





include(FindCUDA)

include_directories(${CUDA_INCLUDE_DIRS})

message(${CUDA_INCLUDE_DIRS})

add_subdirectory( HuygensOnSpeed )
add_subdirectory( HuygensOnSpeedGlutApp )


 
 