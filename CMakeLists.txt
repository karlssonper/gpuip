cmake_minimum_required(VERSION 2.8)

project(gpuip)

if(WIN32)
  set(BUILD_LIB_DEFAULT ON)
else()
  set(BUILD_LIB_DEFAULT OFF)
endif()

option(BUILD_THIRD_PARTY_LIBS "Build needed dependencies if missing" ON)
option(BUILD_SHARED_LIB "Build gpuip as a shared lib" ${BUILD_LIB_DEFAULT})
option(BUILD_WITH_OPENCL "OpenCL" ON)
option(BUILD_WITH_CUDA "CUDA" ON)
option(BUILD_WITH_GLSL "GLSL" ON)
option(BUILD_PYTHON_BINDINGS "PYTHON" ON)
option(BUILD_TESTS "Build test cases" ON)
option(BUILD_DOCS "Build documentation" OFF)

# We use the external project module to build third party libraries
include(ExternalProject)

# Third party repos where the source code is downloaded from
set(GIT_REPO_GLEW https://github.com/karlssonper/glew-cmake.git)
set(GIT_REPO_GLFW https://github.com/glfw/glfw.git)
set(GIT_REPO_BOOST https://github.com/boostorg/boost.git)
set(GIT_REPO_BOOST_NUMPY https://github.com/ndarray/Boost.NumPy.git)
set(GIT_REPO_LIBPNG https://github.com/coapp-packages/libpng.git)
set(GIT_REPO_ZLIB https://github.com/madler/zlib.git)
set(GIT_REPO_OPENEXR https://github.com/karlssonper/openexr.git)
set(GIT_REPO_CIMG https://github.com/hassemlal/CImg.git)

# Set default build type to release (if not specified)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELEASE)
endif()

# Console colors
if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColorReset "${Esc}[m")
  set(Green      "${Esc}[32m")
  set(Yellow     "${Esc}[33m")
  set(Cyan       "${Esc}[36m")
endif()

set(GPUIP_ROOT_DIR ${CMAKE_SOURCE_DIR})

# Local cmake find modules can be found in the cmake folder
set(CMAKE_MODULE_PATH ${GPUIP_ROOT_DIR}/cmake)

# These variables will later be used to link against
set(GPUIP_LIBRARIES)
set(GPUIP_PYTHON_LIBRARIES)

# All third party libraries will be statically built to this dir
set(THIRD_PARTY_DIR ${GPUIP_ROOT_DIR}/3rdparty)

# The gpuip library will depend on these targets
set(THIRD_PARTY_TARGETS)
if(WIN32)
  set(LIB_PREFIX)
  set(LIB_SUFFIX .lib)
else()
  set(LIB_PREFIX lib)
  set(LIB_SUFFIX .a)
endif()

find_package(OpenCL)
if(OPENCL_FOUND AND BUILD_WITH_OPENCL)
  message(STATUS "${Green}Generating build with OpenCL...${ColorReset}")
  include_directories(${OPENCL_INCLUDE_DIRS})
  list(APPEND GPUIP_LIBRARIES ${OPENCL_LIBRARIES})
else()
  message(STATUS "${Yellow}OpenCL disabled .${ColorReset}")
endif()

find_package(CUDA)
if(CUDA_FOUND AND BUILD_WITH_CUDA)
  message(STATUS "${Green}Generating build with CUDA...${ColorReset}")
  find_package(CUDADriver REQUIRED)
  include_directories(${CUDA_INCLUDE_DIRS})
  list(APPEND GPUIP_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_DRIVER_LIBRARY})
  find_package(CUDAMath)
else()
  message(STATUS "${Yellow}CUDA disabled .${ColorReset}")
endif()

find_package(OpenGL)
if(OPENGL_FOUND AND BUILD_WITH_GLSL)
  message(STATUS "${Green}Generating build with GLSL...${ColorReset}")
  include_directories(${OPENGL_INCLUDE_DIRS})
  list(APPEND GPUIP_LIBRARIES ${OPENGL_LIBRARIES})

  # GLFW
  find_package(GLFW)
  if(GLFW_FOUND)
    include_directories(${GLFW_INCLUDE_DIRS})
    list(APPEND GPUIP_LIBRARIES ${GLFW_LIBRARIES})
  elseif(BUILD_THIRD_PARTY_LIBS)
    message(STATUS "${Cyan}Adding third party target glfw${ColorReset}")
    ExternalProject_Add(glfw
      INSTALL_DIR ${THIRD_PARTY_DIR}
      GIT_REPOSITORY ${GIT_REPO_GLFW}
      CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=Release
      -DGLFW_BUILD_EXAMPLES=OFF
      -DGLFW_BUILD_TESTS=OFF
      -DGLFW_BUILD_DOCS=OFF
      -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>)
    list(APPEND THIRD_PARTY_TARGETS glfw)
    list(APPEND GPUIP_LIBRARIES ${LIB_PREFIX}glfw3${LIB_SUFFIX})
  endif()
  if(APPLE)
    list(APPEND GPUIP_LIBRARIES
      "-framework Cocoa" "-framework CoreVideo" "-framework IOKit")
  endif()

  # GLEW
  find_package(GLEW)
  if(GLEW_FOUND)
    include_directories(${GLEW_INCLUDE_DIRS})
    list(APPEND GPUIP_LIBRARIES ${GLEW_LIBRARY})
  elseif(BUILD_THIRD_PARTY_LIBS)
     message(STATUS "${Cyan}Adding third party target glew${ColorReset}")
     ExternalProject_Add(glew
      INSTALL_DIR ${THIRD_PARTY_DIR}
      GIT_REPOSITORY ${GIT_REPO_GLEW}
      CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=Release
      -DONLY_LIBS=TRUE
      -Dglew-cmake_BUILD_SHARED=OFF
      -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>)
    list(APPEND THIRD_PARTY_TARGETS glew)
    list(APPEND GPUIP_LIBRARIES ${LIB_PREFIX}glew${LIB_SUFFIX})
  endif()
else()
  message(STATUS "${Yellow}GLSL disabled .${ColorReset}")
endif()

if(BUILD_PYTHON_BINDINGS)
  message(STATUS "${Green}Generating build with Python bindings..${ColorReset}")
  add_definitions(-D_GPUIP_PYTHON_BINDINGS)

  # Python
  find_package(PythonLibs REQUIRED)
  include_directories(${PYTHON_INCLUDE_DIR})
  list(APPEND GPUIP_PYTHON_LIBRARIES ${PYTHON_LIBRARIES})

  # Boost python
  find_package(Boost COMPONENTS python)
  if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    list(APPEND GPUIP_PYTHON_LIBRARIES ${Boost_PYTHON_LIBRARY})
  elseif(BUILD_THIRD_PARTY_LIBS)
    message(STATUS "${Cyan}Adding third party target boost_python${ColorReset}")
    set(BOOST_GIT_SUBMODULES
      libs/array libs/assert libs/bind libs/concept_check libs/config
      libs/container libs/conversion libs/core libs/detail libs/foreach
      libs/function libs/functional libs/graph libs/integer libs/iterator
      libs/lexical_cast libs/math libs/move libs/mpl libs/multi_index
      libs/numeric/conversion libs/optional libs/parameter libs/predef
      libs/preprocessor libs/property_map libs/python libs/range 
	  libs/serialization libs/smart_ptr libs/static_assert libs/throw_exception
	  libs/tuple libs/type_traits libs/typeof libs/unordered libs/utility 
	  libs/wave tools/build tools/inspect)
    if(WIN32)
      ExternalProject_Add(boost_python
        GIT_REPOSITORY ${GIT_REPO_BOOST}
        #GIT_SUBMODULES ${BOOST_GIT_SUBMODULES}
        BUILD_IN_SOURCE 1
        CONFIGURE_COMMAND bootstrap.bat
        BUILD_COMMAND b2 headers
        INSTALL_COMMAND b2 install --with-python
        link=static runtime-link=static
        variant=release --prefix=${THIRD_PARTY_DIR})
      include_directories(${THIRD_PARTY_DIR}/include/boost-1.57)
      list(APPEND GPUIP_PYTHON_LIBRARIES ${LIB_PREFIX}boost_python-mt${LIB_SUFFIX})
    else()
      ExternalProject_Add(boost_python
        GIT_REPOSITORY ${GIT_REPO_BOOST}
        #GIT_SUBMODULES ${BOOST_GIT_SUBMODULES}
        BUILD_IN_SOURCE 1
        CONFIGURE_COMMAND ./bootstrap.sh
        BUILD_COMMAND ./b2 headers
        INSTALL_COMMAND ./b2 install --with-python
        link=static runtime-link=static
        variant=release --prefix=${THIRD_PARTY_DIR})
      list(APPEND GPUIP_PYTHON_LIBRARIES ${LIB_PREFIX}boost_python${LIB_SUFFIX})
    endif()
    list(APPEND THIRD_PARTY_TARGETS boost_python)
  endif()

  # Boost numpy
  find_package(BoostNumpy)
  if(Boost_NUMPY_FOUND)
    include_directories(${Boost_NUMPY_INCLUDE_DIRS})
    list(APPEND GPUIP_PYTHON_LIBRARIES ${Boost_NUMPY_LIBRARY})
  elseif(BUILD_THIRD_PARTY_LIBS)
    message(STATUS "${Cyan}Adding third party target boost_numpy${ColorReset}")
    if (NOT Boost_FOUND)
      set(BOOST_NUMPY_DEPENDS DEPENDS boost_python)
    endif()
    ExternalProject_Add(boost_numpy
      ${BOOST_NUMPY_DEPENDS}
      INSTALL_DIR ${THIRD_PARTY_DIR}
      GIT_REPOSITORY ${GIT_REPO_BOOST_NUMPY}
      CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=Release
      -DLIBRARY_TYPE=STATIC
      -DBOOST_ROOT=${THIRD_PARTY_DIR}
      -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>)
    list(APPEND GPUIP_PYTHON_LIBRARIES ${LIB_PREFIX}boost_numpy${LIB_SUFFIX})
    list(APPEND THIRD_PARTY_TARGETS boost_numpy)
  endif()

  # ZLIB
  set(ZLIB_DEPENDS)
  find_package(ZLIB)
  if(NOT ZLIB_FOUND)
    message(STATUS "${Cyan}Adding third party target zlib${ColorReset}")
    ExternalProject_Add(zlib
      INSTALL_DIR ${THIRD_PARTY_DIR}
      GIT_REPOSITORY ${GIT_REPO_ZLIB}
      CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>)
    set(ZLIB_DEPENDS DEPENDS zlib)
  else()
    list(APPEND GPUIP_PYTHON_LIBRARIES ${ZLIB_LIBRARIES})
  endif()

  # CImg
  find_path(CIMG_INCLUDE_DIR NAMES CImg.h 
	PATHS ${CIMG_INCLUDE_DIR} $ENV{CIMG_INCLUDE_DIR})
  if(CIMG_INCLUDE_DIR)
	include_directories(${CIMG_INCLUDE_DIR})
  else()
	message(STATUS "${Cyan}Adding third party target cimg${ColorReset}")
	if(WIN32)
	  set(COPY_CMD copy)
	else()
	  set(COPY_CMD cp)
	endif()
    ExternalProject_Add(cimg
      INSTALL_DIR ${THIRD_PARTY_DIR}
      GIT_REPOSITORY ${GIT_REPO_CIMG}
      BUILD_IN_SOURCE 1
	  CONFIGURE_COMMAND "" BUILD_COMMAND "" 
	  INSTALL_COMMAND ${COPY_CMD} CImg.h ${THIRD_PARTY_DIR}/include)
  endif()

  # PNG
  find_package(PNG)
  if(PNG_FOUND)
    include_directories(${PNG_INCLUDE_DIR})
    list(APPEND GPUIP_PYTHON_LIBRARIES ${PNG_LIBRARIES})
  elseif(BUILD_THIRD_PARTY_LIBS)
    message(STATUS "${Cyan}Adding third party target png${ColorReset}")
    ExternalProject_Add(png
      INSTALL_DIR ${THIRD_PARTY_DIR}
      GIT_REPOSITORY ${GIT_REPO_LIBPNG}
      ${ZLIB_DEPENDS}
      CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=Release
      -DPNG_SHARED=OFF
      -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>)
    list(APPEND GPUIP_PYTHON_LIBRARIES ${LIB_PREFIX}png${LIB_SUFFIX})
    list(APPEND THIRD_PARTY_TARGETS png)
  endif()
  add_definitions(-Dcimg_use_png)

  # JPEG
  find_package(JPEG)
  if(JPEG_FOUND)
    add_definitions(-Dcimg_use_jpeg)
    include_directories(${JPEG_INCLUDE_DIR})
    list(APPEND GPUIP_PYTHON_LIBRARIES ${JPEG_LIBRARIES})
  endif()

  # TIFF
  find_package(TIFF)
  if(TIFF_FOUND)
    add_definitions(-Dcimg_use_tiff)
    include_directories(${TIFF_INCLUDE_DIR})
    list(APPEND GPUIP_PYTHON_LIBRARIES ${TIFF_LIBRARIES})
  endif()

  # OpenEXR
  find_package(OpenEXR)
  if(OPENEXR_FOUND)
    include_directories(${OPENEXR_INCLUDE_PATHS})
    list(APPEND GPUIP_PYTHON_LIBRARIES ${OPENEXR_LIBRARIES})
  elseif(BUILD_THIRD_PARTY_LIBS)
    message(STATUS "${Cyan}Adding third party target openexr${ColorReset}")
    ExternalProject_Add(openexr
      INSTALL_DIR ${THIRD_PARTY_DIR}
      GIT_REPOSITORY ${GIT_REPO_OPENEXR}
      ${ZLIB_DEPENDS}
      CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=Release
      -DZLIB_ROOT=${THIRD_PARTY_DIR}
      -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
      -DBUILD_SHARED_LIBS=OFF
      -DBUILD_BIN=OFF
      -DBUILD_TESTS=OFF)
    list(APPEND GPUIP_PYTHON_LIBRARIES ${LIB_PREFIX}Half${LIB_SUFFIX})
    set(OPENEXR_LIBS Iex IexMath IlmImf IlmThread Imath)
    set(VER -2_1)
    foreach(_L ${OPENEXR_LIBS})
      list(APPEND GPUIP_PYTHON_LIBRARIES ${LIB_PREFIX}${_L}${VER}${LIB_SUFFIX})
    endforeach()
    list(APPEND THIRD_PARTY_TARGETS openexr)
  endif()

  add_subdirectory(python)
else()
  message(STATUS "${Yellow}Python bindings disabled .${ColorReset}")
endif()

if (THIRD_PARTY_TARGETS)
  include_directories(${THIRD_PARTY_DIR}/include)
  link_directories(${THIRD_PARTY_DIR}/lib)
endif()
add_subdirectory(src)

if(BUILD_TESTS)
  message(STATUS "Testing enabled")
  enable_testing()

  # Add C++ test
  add_executable(test_cpp test/test)
  target_link_libraries(test_cpp gpuip)
  add_test(NAME test_cpp COMMAND test_cpp)

  # Add python test
  if(BUILD_PYTHON_BINDINGS)
    find_package(PythonInterp REQUIRED)
    configure_file(test/test.py src/test.py COPYONLY)
    add_test(NAME test_py COMMAND ${PYTHON_EXECUTABLE} src/test.py)
  endif()
endif()

if(BUILD_DOCS)
  add_subdirectory(doc)
endif()



