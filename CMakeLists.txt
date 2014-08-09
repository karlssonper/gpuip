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

macro(_build_from_submodule submodule_name cmake_dir cmake_args)
  execute_process(COMMAND git submodule init 3rdparty/${submodule_name} 
	WORKING_DIRECTORY ${GPUIP_ROOT_DIR})
  execute_process(COMMAND git submodule update 3rdparty/${submodule_name} 
	WORKING_DIRECTORY ${GPUIP_ROOT_DIR})
  set(BUILD_DIR ${GPUIP_ROOT_DIR}/3rdparty/${submodule_name}/${cmake_dir}/build)
  file(MAKE_DIRECTORY ${BUILD_DIR})
  find_program(cmake_bin NAMES cmake)
  message("command " "${cmake_bin}" "${cmake_args}" "..")
  add_custom_target(
	generate_${submodule_name}_${cmake_dir} 
	COMMAND "${cmake_bin}"  "${cmake_args}" ".."  VERBATIM
	WORKING_DIRECTORY ${BUILD_DIR})
  add_custom_target(
	build_${submodule_name}_${cmake_dir} 
	COMMAND "${cmake_bin} " --build . --config Release --target INSTALL
	WORKING_DIRECTORY ${BUILD_DIR})
  add_dependencies(build_${submodule_name}_${cmake_dir} generate_${submodule_name}_${cmake_dir})
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
	find_package(ZLIB)
	#if (WIN32)
	#  add_definitions(-DPLATFORM_WINDOWS)
	#endif(WIN32)
	#set(CMAKE_SOURCE_DIR ${GPUIP_ROOT_DIR}/3rdparty/openexr/IlmBase)
	set(ILMBASE_ARGS -DBUILD_SHARED_LIBS=OFF)
	_build_from_submodule(openexr IlmBase ${ILMBASE_ARGS})
	#set(IlmBaseLibs Half Iex IlmThread Imath)
	#include_directories(${GPUIP_ROOT_DIR}/3rdparty/openexr/IlmBase/config)
	#include_directories(${CMAKE_CURRENT_BINARY_DIR}/3rdparty/zlib)
	#set(ILM_BASE_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/3rdparty/openexr/IlmBase)
	#foreach(ILM_LIB ${IlmBaseLibs})
	# include_directories(${GPUIP_ROOT_DIR}/3rdparty/openexr/IlmBase/${ILM_LIB})
	#  link_directories(${ILM_BASE_BUILD_DIR}/${ILM_LIB}/Release)
	#endforeach()
	#set(CMAKE_SOURCE_DIR ${GPUIP_ROOT_DIR}/3rdparty/openexr/OpenEXR)
	#configure_file(${ILM_BASE_BUILD_DIR}/Half/Release/Half.dll ${CMAKE_SOURCE_DIR}/IlmImf/Half.dll)
	#configure_file(${ILM_BASE_BUILD_DIR}/Iex/Release/Iex-2_1.dll ${CMAKE_SOURCE_DIR}/IlmImf/Iex-2_1.dll)
	#configure_file(${ILM_BASE_BUILD_DIR}/IlmThread/Release/IlmThread-2_1.dll ${CMAKE_SOURCE_DIR}/IlmImf/IlmThread-2_1.dll)
	set(OPENEXR_ARGS "-DZLIB_ROOT=${GPUIP_ROOT_DIR}/3rdparty/zlib")
	_build_from_submodule(openexr OpenEXR ${OPENEXR_ARGS})

	if (NOT ZLIB_FOUND)
	  _build_from_submodule(zlib "" "")
	  add_dependencies(build_openexr_OpenEXR build_zlib_ build_openexr_IlmBase)
	  #set(ZLIB_ROOT  ${GPUIP_ROOT_DIR}/3rdparty/zlib)
	  configure_file(${GPUIP_ROOT_DIR}/3rdparty/zlib/zlib.h 
	    ${GPUIP_ROOT_DIR}/3rdparty/zlib/include/zlib.h)
	  write_file(${GPUIP_ROOT_DIR}/3rdparty/zlib/lib/zlib.lib "tmp")
	endif(NOT ZLIB_FOUND)

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