# Version 2.8 req since it includes the find cuda module
cmake_minimum_required(VERSION 2.8)

project(gpuip)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Common variables for compling the library
set(SOURCE src/gpuip)
set(LIBRARIES)

# Build with OpenCL
option(OpenCL "OpenCL" ON)
if(OpenCL)
  message("-- Generating build with OpenCL...")

  find_package(OpenCL REQUIRED)
  include_directories(${OPENCL_INCLUDE_DIRS})
  set(LIBRARIES ${LIBRARIES} ${OPENCL_LIBRARIES})

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
  include_directories(${CUDA_INCLUDE_DIRS})
  
  set(CUDA_DRIVER_LIBRARY ${CUDA_LIBRARIES})
  STRING(REGEX REPLACE "cudart" "cuda" CUDA_DRIVER_LIBRARY ${CUDA_DRIVER_LIBRARY})

  message(${CUDA_LIBRARIES})
  # Link against the Cuda libraries 
  set(LIBRARIES ${LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_DRIVER_LIBRARY})

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
  
  # Easier to set these manually in windows
  if (WIN32)
	set(GLEW_INCLUDE_DIRS $ENV{GLEW_INCLUDE_DIR})
	set(GLEW_LIBRARY $ENV{GLEW_LIBRARY_DIR}/glew32.lib)
  else (WIN32)
	find_package(GLEW REQUIRED)
  endif(WIN32)
  find_package(GLFW REQUIRED)
  include_directories(${OPENGL_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS} ${GLFW_INCLUDE_DIRS})  

  set(SOURCE ${SOURCE} src/glsl) 
  set(LIBRARIES ${LIBRARIES} ${OPENGL_LIBRARIES} ${GLFW_LIBRARIES} ${GLEW_LIBRARY})
else(GLSL)
  message("-- GLSL disabled .")
endif(GLSL)

add_library(gpuip ${SOURCE})
target_link_libraries(gpuip ${LIBRARIES})

#set(BOOST_ROOT "C:/Boost/include/boost-1_45")
#set(Boost_NO_SYSTEM_PATHS true)
#set(BOOST_INCLUDEDIR "C:/Boost/include/boost-1_45/")
#set(BOOST_LIBRARYDIR "C:/Boost/lib")
#set(Boost_USE_MULTITHREAD ON)
#SET(Boost_USE_STATIC_LIBS OFF)
#SET(Boost_USE_MULTITHREADED ON)		#
#SET(Boost_USE_STATIC_RUNTIME OFF)
#add_definitions(-DBOOST_ALL_NO_LIB)
#add_definitions(-DBOOST_PYTHON_STATIC_LIB)
# Build python bindings (using boost python)
option(PYTHON "PYTHON" ON)
if(PYTHON)
  message("-- Generating build with python bindings...")
  find_package(Boost REQUIRED COMPONENTS python)
  find_package(PythonLibs)
  include_directories(${PYTHON_INCLUDE_DIR} ${Boost_INCLUDE_DIRS})
  # Preprocessor will use boost shared pointers instead of std
  add_definitions(-D_GPUIP_PYTHON_BINDINGS)

  # compiler definitions for non-windows builds
  if (NOT WIN32)
	add_definitions(-fPIC)
  endif()
  add_library(pyGpuip SHARED src/python.cpp )
  set_target_properties( pyGpuip PROPERTIES PREFIX "")
  if (WIN32)
	set_target_properties( pyGpuip PROPERTIES SUFFIX ".pyd")
  else (WIN32)
	set_target_properties( pyGpuip PROPERTIES SUFFIX ".so")
  endif()
 
  #link_directories(${Boost_LIBRARY_DIRS})
  message(${Boost_PYTHON_LIBRARY} )
foreach( test  ${BOOST_LIBRARIES})
    message("BOOST LIBRARIES: " ${test})
  endforeach( test   ${BOOST_LIBRARIES})
  set(Boost_NUMPY_LIBRARY ${Boost_LIBRARY_DIRS}/boost_numpy.lib)
  message(${Boost_NUMPY_LIBRARY})

  #set(BOOST_LIBRARIES ${BOOST_LIBRARIES} boost_numpy)
  #target_link_libraries(pyGpuip gpuip ${PYTHON_LIBRARIES} "C:/Boost/lib/boost_python-vc100-mt-1_55.lib"  "C:/Boost/lib/libboost_python-vc100-mt-1_55.lib" "C:/Boost/lib/boost_numpy.lib")
  target_link_libraries(pyGpuip gpuip ${PYTHON_LIBRARIES} ${Boost_PYTHON_LIBRARY} ${Boost_NUMPY_LIBRARY})

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
