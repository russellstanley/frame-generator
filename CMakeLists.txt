# Init CMake (require at least version 3.16)
CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

# Project name and version
PROJECT(frame-generator
	VERSION 1.0.0
	LANGUAGES C CXX)

FIND_PACKAGE(dv 1.5.0 REQUIRED)

# Compile this module (standard C++ file, no additional dependencies)
ADD_LIBRARY(frame_generator SHARED FrameGenerator.cpp)

SET_TARGET_PROPERTIES(frame_generator
	PROPERTIES
	PREFIX "user_"
)

TARGET_LINK_LIBRARIES(frame_generator PRIVATE dv::sdk)

INSTALL(TARGETS frame_generator DESTINATION ${DV_MODULES_DIR})
