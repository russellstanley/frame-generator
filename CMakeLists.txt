# Init CMake (require at least version 3.16)
CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

# Project name and version
PROJECT(hats
	VERSION 1.0.0
	LANGUAGES C CXX)

FIND_PACKAGE(dv 1.5.0 REQUIRED)

# Compile this module (standard C++ file, no additional dependencies)
ADD_LIBRARY(hats SHARED HatsGenerator.cpp)

SET_TARGET_PROPERTIES(hats
	PROPERTIES
	PREFIX "user_"
)

TARGET_LINK_LIBRARIES(hats PRIVATE dv::sdk)

INSTALL(TARGETS hats DESTINATION ${DV_MODULES_DIR})
