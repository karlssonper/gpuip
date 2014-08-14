# Version 2.8 req since it includes the find cuda module
cmake_minimum_required(VERSION 2.8)

project(gpuip)

option(BUILD_THIRD_PARTY_LIBS "Build needed dependencies if missing" ON)
option(BUILD_SHARED_LIB "Build shared lib" OFF)
option(OPENCL "OpenCL" ON)
option(CUDA "CUDA" ON)
option(GLSL "GLSL" ON)
option(PYTHON "PYTHON" ON)
option(TEST "TEST" ON)
option(BUILD_DOCS "Build documentation" OFF)

set(GPUIP_ROOT_DIR ${CMAKE_SOURCE_DIR})

# Set default build type to release (if not specified)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELEASE)
endif()

# Local cmake find modules can be found in the cmake folder
set(CMAKE_MODULE_PATH ${GPUIP_ROOT_DIR}/cmake)

set(GPUIP_LIBRARIES)
set(GPUIP_PYTHON_LIBRARIES)
set(MISSING_THIRD_PARTY)

if(OPENCL)
  find_package(OpenCL REQUIRED)
  include_directories(${OPENCL_INCLUDE_DIRS})
  list(APPEND GPUIP_LIBRARIES ${OPENCL_LIBRARIES})
endif(OPENCL)

if(CUDA)
  find_package(CUDA REQUIRED)
  find_package(CUDADriver REQUIRED)
  include_directories(${CUDA_INCLUDE_DIRS})
  list(APPEND GPUIP_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_DRIVER_LIBRARY})
  find_package(CUDAMath)
endif(CUDA)

if(GLSL)
  # OpenGL Core
  find_package(OpenGL REQUIRED)
  include_directories(${OPENGL_INCLUDE_DIRS})
  list(APPEND GPUIP_LIBRARIES ${OPENGL_LIBRARIES})

  # GLFW
  find_package(GLFW)
  if (GLFW_FOUND)
	include_directories(${GLFW_INCLUDE_DIRS})
	list(APPEND GPUIP_LIBRARIES ${GLFW_LIBRARIES})
  else (GLFW_FOUND)
	list(APPEND MISSING_THIRD_PARTY glfw)
  endif (GLFW_FOUND)
  if (APPLE)
	list(APPEND GPUIP_LIBRARIES  
	  "-framework Cocoa" "-framework CoreVideo" "-framework IOKit")
  endif (APPLE)

  # GLEW
  find_package(GLEW)
  if (GLEW_FOUND)
	include_directories(${GLEW_INCLUDE_DIRS})
	list(APPEND GPUIP_LIBRARIES ${GLEW_LIBRARY})
  else (GLEW_FOUND)
	list(APPEND MISSING_THIRD_PARTY glew)
  endif (GLEW_FOUND)
endif(GLSL)

if (PYTHON)
  add_definitions(-D_GPUIP_PYTHON_BINDINGS)
  
  # Python
  find_package(PythonLibs REQUIRED)
  include_directories(${PYTHON_INCLUDE_DIR})
  list(APPEND GPUIP_PYTHON_LIBRARIES ${PYTHON_LIBRARIES})

  # Boost python
  find_package(Boost COMPONENTS python)
  if (Boost_FOUND)
	include_directories(${Boost_INCLUDE_DIRS})
	list(APPEND GPUIP_PYTHON_LIBRARIES ${Boost_PYTHON_LIBRARY})
  else (Boost_FOUND)
	list(APPEND MISSING_THIRD_PARTY boost_python)
  endif (Boost_FOUND)

  # Boost numpy
  find_package(BoostNumpy)
  if (Boost_NUMPY_FOUND)
	include_directories(${Boost_NUMPY_INCLUDE_DIRS})
	list(APPEND GPUIP_PYTHON_LIBRARIES ${Boost_NUMPY_LIBRARY})
  else (Boost_NUMPY_FOUND)
	list(APPEND MISSING_THIRD_PARTY boost_numpy)
  endif (Boost_NUMPY_FOUND)
	
  # PNG
  find_package(PNG)
  if (PNG_FOUND)
	add_definitions(-Dcimg_use_png)
	include_directories(${PNG_INCLUDE_DIR})
	list(APPEND GPUIP_PYTHON_LIBRARIES ${PNG_LIBRARIES})
  else (PNG_FOUND)
	list(APPEND MISSING_THIRD_PARTY png)
  endif (PNG_FOUND)

  # JPEG
  find_package(JPEG)
  if(JPEG_FOUND)
	add_definitions(-Dcimg_use_jpeg)
	include_directories(${JPEG_INCLUDE_DIR})
	list(APPEND GPUIP_PYTHON_LIBRARIES ${JPEG_LIBRARIES})
  endif(JPEG_FOUND)

  # TIFF
  find_package(TIFF)
  if(TIFF_FOUND)
	add_definitions(-Dcimg_use_tiff)
	include_directories(${TIFF_INCLUDE_DIR})
	list(APPEND GPUIP_PYTHON_LIBRARIES ${TIFF_LIBRARIES})
  endif(TIFF_FOUND)

  # OpenEXR
  find_package(OpenEXR)
  if (OPENEXR_FOUND)
	include_directories(${OPENEXR_INCLUDE_PATHS})
	list(APPEND GPUIP_PYTHON_LIBRARIES ${OPENEXR_LIBRARIES})
  else (OPENEXR_FOUND)
	find_package(ZLIB)
	if (NOT ZLIB_FOUND)
	  list(APPEND MISSING_THIRD_PARTY zlib)
	endif(NOT ZLIB_FOUND)
	list(APPEND MISSING_THIRD_PARTY openexr)
  endif (OPENEXR_FOUND)
endif(PYTHON)

include(ExternalProject)
ExternalProject_Add(glew
  INSTALL_DIR ${GPUIP_ROOT_DIR}/3rdparty/temp_install
  GIT_REPOSITORY https://github.com/karlssonper/glew-cmake
  CMAKE_ARGS 
  -DONLY_LIBS=TRUE 
  -Dglew-cmake_BUILD_SHARED=OFF
  -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>)

ExternalProject_Add(glfw
  INSTALL_DIR ${GPUIP_ROOT_DIR}/3rdparty/temp_install
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  CMAKE_ARGS 
  -DGLFW_BUILD_EXAMPLES=OFF
  -DGLFW_BUILD_TESTS=OFF
  -DGLFW_BUILD_DOCS=OFF
  -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>)

ExternalProject_Add(zlib
  INSTALL_DIR ${GPUIP_ROOT_DIR}/3rdparty/temp_install
  GIT_REPOSITORY https://github.com/madler/zlib.git
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>)

ExternalProject_Add(png
  INSTALL_DIR ${GPUIP_ROOT_DIR}/3rdparty/temp_install
  GIT_REPOSITORY https://github.com/coapp-packages/libpng.git
  CMAKE_ARGS 
  -DPNG_SHARED=OFF
  -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>)

ExternalProject_Add(openexr
  INSTALL_DIR ${GPUIP_ROOT_DIR}/3rdparty/temp_install
  GIT_REPOSITORY https://github.com/karlssonper/openexr
  CMAKE_ARGS
  -DZLIB_ROOT=${GPUIP_ROOT_DIR}/3rdparty/temp_install
  -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
  -DBUILD_SHARED_LIBS=OFF
  -DBUILD_BIN=OFF
  -DBUILD_TESTS=OFF)



set(BOOST_GIT_SUBMODULES
  libs/array libs/assert libs/bind libs/concept_check libs/config 
  libs/container libs/conversion libs/core libs/detail libs/foreach 
  libs/function libs/functional libs/graph libs/integer libs/iterator 
  libs/lexical_cast libs/math libs/move libs/mpl libs/multi_index 
  libs/numeric/conversion libs/optional libs/parameter libs/predef 
  libs/preprocessor libs/property_map libs/python libs/range libs/serialization 
  libs/smart_ptr libs/static_assert libs/throw_exception libs/tuple 
  libs/type_traits libs/typeof libs/unordered libs/utility libs/wave 
  tools/build tools/inspect)
if (WIN32)
  ExternalProject_Add(boost_python
  BUILD_IN_SOURCE 1
  CONFIGURE_COMMAND bootstrap.bat
  BUILD_COMMAND b2 headers
  INSTALL_COMMAND b2 install --with-python --variant=release --prefix=${GPUIP_ROOT_DIR}/3rdparty/temp_install 
  GIT_REPOSITORY https://github.com/boostorg/boost
  GIT_SUBMODULES ${BOOST_GIT_SUBMODULES})
else (WIN32)
  ExternalProject_Add(boost_python
  BUILD_IN_SOURCE 1
  CONFIGURE_COMMAND ./bootstrap.sh
  BUILD_COMMAND ./b2 headers
  INSTALL_COMMAND ./b2 install --with-python --variant=release --prefix=${GPUIP_ROOT_DIR}/3rdparty/temp_install
  GIT_REPOSITORY https://github.com/boostorg/boost
  GIT_SUBMODULES ${BOOST_GIT_SUBMODULES})
endif (WIN32)



ExternalProject_Add(boost_numpy
  DEPENDS boost_python
  INSTALL_DIR ${GPUIP_ROOT_DIR}/3rdparty/temp_install
  GIT_REPOSITORY https://github.com/ndarray/Boost.NumPy.git
  CMAKE_ARGS
  -DLIBRARY_TYPE=STATIC
  -DBOOST_ROOT=${GPUIP_ROOT_DIR}/3rdparty/temp_install
  -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>)

if (BUILD_THIRD_PARTY_LIBS AND MISSING_THIRD_PARTY)
  set(TEMP_INSTALL_DIR ${GPUIP_ROOT_DIR}/3rdparty/temp_install)
  include_directories(${TEMP_INSTALL_DIR}/include)
  #add_custom_target(third_party 
  #COMMAND python build_3rdparty.py 
  #--install-prefix ${TEMP_INSTALL_DIR}
  #${MISSING_THIRD_PARTY}
  #WORKING_DIRECTORY ${GPUIP_ROOT_DIR})

  if (GLSL)
	if (NOT GLFW_FOUND)
	  find_library(GLFW_LIBRARY NAMES glfw3 PATHS ${TEMP_INSTALL_DIR}/lib)
	  list(APPEND GPUIP_LIBRARIES ${GLFW_LIBRARY})
	endif (NOT GLFW_FOUND)
	if (NOT GLEW_FOUND)
	  find_library(GLEW_LIBRARY NAMES glew PATHS ${TEMP_INSTALL_DIR}/lib)
	  list(APPEND GPUIP_LIBRARIES ${GLEW_LIBRARY})
	endif (NOT GLEW_FOUND)
  endif (GLSL)

  if (PYTHON)
	if (NOT Boost_FOUND)
	  find_library(BP_LIBRARY NAMES boost_python PATHS ${TEMP_INSTALL_DIR}/lib)
	  list(APPEND GPUIP_PYTHON_LIBRARIES ${BP})
	endif (NOT Boost_FOUND)
	if (NOT Boost_NUMPY_FOUND)
	  find_library(BPN_LIBRARY NAMES boost_numpy PATHS ${TEMP_INSTALL_DIR}/lib)
	  list(APPEND GPUIP_PYTHON_LIBRARIES ${BPN})
	endif (NOT Boost_NUMPY_FOUND)
	
  endif (PYTHON)

  install(CODE "execute_process(COMMAND python build_3rdparty.py 
	--install-prefix ${CMAKE_INSTALL_PREFIX}
	${MISSING_THIRD_PARTY}
	WORKING_DIRECTORY ${GPUIP_ROOT_DIR})")
endif (BUILD_THIRD_PARTY_LIBS AND MISSING_THIRD_PARTY)

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