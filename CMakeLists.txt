# Version 2.8 req since it includes the find cuda module
cmake_minimum_required(VERSION 2.8)

project(gpuip)

option(BUILD_SHARED_LIB "Build shared lib" OFF)
option(OPENCL "OpenCL" ON)
option(CUDA "CUDA" ON)
option(GLSL "GLSL" ON)
option(PYTHON "PYTHON" ON)
option(TEST "TEST" ON)
option(BUILD_DOCS "Build documentation" OFF)

set(GPUIP_ROOT_DIR ${CMAKE_SOURCE_DIR})

macro(_build_from_submodule submodule_name cmake_dir)
  execute_process(COMMAND git submodule init 3rdparty/${submodule_name} 
	WORKING_DIRECTORY ${GPUIP_ROOT_DIR})
  execute_process(COMMAND git submodule update 3rdparty/${submodule_name} 
	WORKING_DIRECTORY ${GPUIP_ROOT_DIR})
  add_subdirectory(${GPUIP_ROOT_DIR}/3rdparty/${submodule_name}/${cmake_dir})
endmacro()

# Set default build type to release (if not specified)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELEASE)
endif()

# Local cmake find modules can be found in the cmake folder
set(CMAKE_MODULE_PATH ${GPUIP_ROOT_DIR}/cmake)

if(OPENCL)
  find_package(OpenCL REQUIRED)
endif(OPENCL)

if(CUDA)
  find_package(CUDA REQUIRED)
  find_package(CUDADriver REQUIRED)
  find_package(CUDAMath)
endif(CUDA)

if(GLSL)
  find_package(OpenGL REQUIRED)   
  find_package(GLFW REQUIRED)
  if (NOT WIN32)
	find_package(GLEW REQUIRED)
  endif()

endif(GLSL)

if (PYTHON)
  add_definitions(-D_GPUIP_PYTHON_BINDINGS)
  find_package(Boost REQUIRED COMPONENTS python)
  find_package(BoostNumpy REQUIRED)
  find_package(PythonLibs REQUIRED)
  find_package(PNG)
  find_package(TIFF)
  find_package(JPEG)
  find_package(OpenEXR)

  if(NOT OPENEXR_FOUND)
	message(STATUS "Building openexr from source")
	if (WIN32)
	  add_definitions(-DPLATFORM_WINDOWS)
	endif(WIN32)
	set(CMAKE_SOURCE_DIR ${GPUIP_ROOT_DIR}/3rdparty/openexr/IlmBase)
	_build_from_submodule(openexr IlmBase)
	#add_subdirectory(${GPUIP_ROOT_DIR}/3rdparty/openexr/OpenEXR)
  endif(NOT OPENEXR_FOUND)

  add_subdirectory(python)
endif(PYTHON)

#add_subdirectory(src)

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