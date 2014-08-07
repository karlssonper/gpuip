# Version 2.8 req since it includes the find cuda module
cmake_minimum_required(VERSION 2.8)

project(gpuip)

# Set default build type to release (if not specified)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELEASE)
endif()

# Local cmake find modules can be found in the cmake folder
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

option(BUILD_SHARED_LIB "Build shared lib" OFF)
option(OPENCL "OpenCL" ON)
option(CUDA "CUDA" ON)
option(GLSL "GLSL" ON)
option(PYTHON "PYTHON" ON)
option(TEST "TEST" ON)
option(BUILD_DOCS "Build documentation" OFF)

if (PYTHON)
  add_definitions(-D_GPUIP_PYTHON_BINDINGS)
  add_subdirectory(python)
endif(PYTHON)

add_subdirectory(src)

if (TEST)
  enable_testing()
  message(STATUS "Testing enabled")

  # Add C++ test
  add_executable(test_cpp test/test)
  target_link_libraries(test_cpp gpuip)
  add_test(NAME test_cpp COMMAND test_cpp)

  # Add python test
  if (PYTHON)
	find_package(PythonInterp REQUIRED)
	configure_file(test/test.py src/test.py COPYONLY)
	add_test(NAME test_py COMMAND ${PYTHON_EXECUTABLE} src/test.py)
  endif(PYTHON)
endif(TEST)

if (BUILD_DOCS)
  add_subdirectory(doc)
endif(BUILD_DOCS)