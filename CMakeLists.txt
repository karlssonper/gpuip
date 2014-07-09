# Version 2.8 req since it includes the find cuda module
cmake_minimum_required(VERSION 2.8)

project(gpuip)

# Common variables for compling the library
set(SOURCE src/gpuip)
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

  find_package(OpenGL REQUIRED)   
  find_package(GLUT REQUIRED)
  find_package(GLEW REQUIRED)

  include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})  

  set(SOURCE ${SOURCE} src/glsl) 
  set(LIBRARIES ${LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${GLEW_LIBRARY})
else(GLSL)
  message("-- GLSL disabled .")
endif(GLSL)

add_library(gpuip ${SOURCE})
target_link_libraries(gpuip ${LIBRARIES})

# Build python bindings (using boost python)
option(PYTHON "PYTHON" ON)
if(PYTHON)
  message("-- Generating build with python bindings...")
  find_package(Boost 1.45.0 COMPONENTS python)
  find_package(PythonLibs)
  include_directories(${PYTHON_INCLUDE_DIR})
  #include_directories(/Users/per/Downloads/boost_1_55_0)
  #add_library(PyIp SHARED src/python.cpp )
  #LINK_DIRECTORIES(/Users/per/Downloads/boost_1_55_0/stage/lib)

  # Preprocessor will use boost shared pointers instead of std
  add_definitions(-D_GPUIP_PYTHON_BINDINGS)

  # compiler definitions for non-windows builds
  if (NOT WIN32)
	add_definitions(-fPIC)
  endif()
  add_library(pyGpuip SHARED src/python.cpp )
  set_target_properties( pyGpuip PROPERTIES PREFIX "")
  set_target_properties( pyGpuip PROPERTIES SUFFIX ".so")
  target_link_libraries(pyGpuip gpuip ${PYTHON_LIBRARIES} boost_python boost_numpy)

else(PYTHON)
  message("-- Python bindings disabled.")
endif(PYTHON)

# Enable unit testing
enable_testing()

# Add C++ test
add_executable(test_cpp test/test)
target_link_libraries(test_cpp gpuip)
add_test(NAME test_cpp COMMAND test_cpp)

# Add python test
if (PYTHON)
  find_package(PythonInterp REQUIRED)
  configure_file(test/test.py test.py COPYONLY)
  add_test(NAME test_py COMMAND ${PYTHON_EXECUTABLE} test.py)
endif(PYTHON)
