
add_executable(HuygensOnSpeedApp HuygensOnSpeed.cpp)

if ( UNIX )
   target_link_libraries( HuygensOnSpeedApp HuygensOnSpeed glut ${GLEW_LIBRARIES} GL)
else ()
   if (CMAKE_CL_64)
      target_link_libraries( HuygensOnSpeedApp HuygensOnSpeed freeglut glew64)
   else ()
      target_link_libraries( HuygensOnSpeedApp HuygensOnSpeed freeglut glew32)
   endif ()
endif ()
