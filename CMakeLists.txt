# Version 2.8 req since it includes the find cuda module
cmake_minimum_required(VERSION 2.8)

project(gpuip)

# Local cmake find modules can be found in the cmake folder
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

option(OpenCL "OpenCL" ON)
option(CUDA "CUDA" ON)
option(GLSL "GLSL" ON)
option(PYTHON "PYTHON" ON)
option(TEST "TEST" ON)

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
  target_link_libraries(test_cpp gpuip_static)
  add_test(NAME test_cpp COMMAND test_cpp)

  # Add python test
  if (PYTHON)
	find_package(PythonInterp REQUIRED)
	configure_file(test/test.py test.py COPYONLY)
	add_test(NAME test_py COMMAND ${PYTHON_EXECUTABLE} test.py)
  endif(PYTHON)
endif(TEST)
