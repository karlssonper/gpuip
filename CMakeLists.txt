# Version 2.8 req as it includes the find cuda module
cmake_minimum_required(VERSION 2.8)

project(gpuip)

# Common variables for compling the library
set(SOURCE src/base src/gpuip src/cuda src/glsl)
set(LIBRARIES)

# Build with OpenCL
option(OpenCL "OpenCL" ON)
if(OpenCL)
  message("-- Generating build with OpenCL...\n")
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
  message("-- OpenCL disabled .\n")
endif(OpenCL)

add_library(gpuip ${SOURCE})
target_link_libraries(gpuip ${LIBRARIES})

# Add unit tests
enable_testing()
add_executable(test_cpp test/test)
target_link_libraries(test_cpp gpuip)
add_test(NAME testcpp
         COMMAND testcpp)
