

add_executable(HuygensOnSpeedApp HuygensOnSpeed.cpp)

target_link_libraries( HuygensOnSpeedApp HuygensOnSpeed glut ${GLEW_LIBRARIES})
# target_link_libraries( HuygensOnSpeedApp HuygensOnSpeed glut)