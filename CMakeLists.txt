# Version 2.8 req since it includes the find cuda module
cmake_minimum_required(VERSION 2.8)

project(gpuip)

# Common variables for compling the library
set(SOURCE src/gpuip src/base)
set(LIBRARIES)

# Build with OpenCL
option(OpenCL "OpenCL" ON)
if(OpenCL)
  message("-- Generating build with OpenCL...")
  if (APPLE)
     find_library(OPENCL_LIBRARY OpenCL DOC "OpenCL lib for OSX")
     link_directories(${OPENCL_LIBRARY})
	 find_path(OPENCL_INCLUDE_DIR OpenCL/cl.h DOC "Include for OpenCL on OSX")
     set(LIBRARIES ${LIBRARIES} ${OPENCL_LIBRARY})
   else(APPLE)
	 # Use the environment variables OPENCL_INCLUDE_DIR to find OpenCL headers
     include_directories($ENV{OPENCL_INCLUDE_DIR})  

	 # Use the environment variable OPENCL_LIB_DIR to find the OpenCl so files
	 link_directories($ENV{OPENCL_LIB_DIR})

	 # Use libOpenCL when linking later on
     set(LIBRARIES ${LIBRARIES} OpenCL)
  endif(APPLE)

  # Build the OpenCLImpl
  set(SOURCE ${SOURCE} src/opencl)  
else(OpenCL)
  message("-- OpenCL disabled .")
endif(OpenCL)

# Build with CUDA
option(CUDA "CUDA" ON)
if(CUDA)
  message("-- Generating build with CUDA...")

  # Use standard CMake find cuda module
  find_package(CUDA REQUIRED)

  # Add cuda include directories
  include_directories(${CUDA_INCLUDE})
  
  # Link against the Cuda libraries 
  set(LIBRARIES ${LIBRARIES} cuda cudart)

  # Build the CUDAImpl
  set(SOURCE ${SOURCE} src/cuda)  
else(CUDA)
  message("-- CUDA disabled .")
endif(CUDA)

# Build with GLSL (needs OpenGL)
option(GLSL "GLSL" ON)
if(GLSL)
  message("-- Generating build with GLSL...")
  set(SOURCE ${SOURCE} src/glsl)  
else(GLSL)
  message("-- GLSL disabled .")
endif(GLSL)

add_library(gpuip ${SOURCE})
target_link_libraries(gpuip ${LIBRARIES})

# Add unit tests
enable_testing()
add_executable(test_cpp test/test)
target_link_libraries(test_cpp gpuip)
add_test(NAME testcpp COMMAND testcpp)
