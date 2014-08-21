cmake_minimum_required(VERSION 2.8)

project(gpuip)

# Specific versions used
set(BOOST_VERSION 1.56)
set(OPENEXR_VERSION 2.1)

if(WIN32)
	set(BUILD_LIB_DEFAULT OFF)
else()
	set(BUILD_LIB_DEFAULT ON)
endif()

option(BUILD_THIRD_PARTY_LIBS "Build needed dependencies if missing" ON)
option(BUILD_SHARED_LIB "Build gpuip as a shared lib" ${BUILD_LIB_DEFAULT})
option(BUILD_WITH_OPENCL "OpenCL" ON)
option(BUILD_WITH_CUDA "CUDA" ON)
option(BUILD_WITH_GLSL "GLSL" ON)
option(BUILD_PYTHON_BINDINGS "PYTHON" ON)
option(BUILD_TESTS "Build test cases" ON)
option(BUILD_DOCS "Build documentation" OFF)
option(DOWNLOAD_EXAMPLES_IMAGES "Download examples input images" ON)

# We use the external project module to build third party libraries
include(ExternalProject)

# Third party repos where the source code is downloaded from
set(GIT_REPO_GLEW https://github.com/omniavinco/glew-cmake.git)
set(GIT_REPO_GLFW https://github.com/glfw/glfw.git)
set(GIT_REPO_BOOST https://github.com/boostorg/boost.git)
set(GIT_REPO_BOOST_NUMPY https://github.com/karlssonper/Boost.NumPy)
set(GIT_REPO_LIBPNG https://github.com/coapp-packages/libpng.git)
set(GIT_REPO_ZLIB https://github.com/madler/zlib.git)
set(GIT_REPO_OPENEXR https://github.com/karlssonper/openexr.git)
set(GIT_REPO_CIMG https://github.com/hassemlal/CImg.git)
set(GIT_REPO_EXAMPLES_IMAGES https://github.com/karlssonper/gpuip-examples-images.git)

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
set(GPUIP_TEST_LIBRARIES)

# All third party libraries will be statically built to this dir
set(THIRD_PARTY_DIR ${GPUIP_ROOT_DIR}/thirdparty)

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
			-DCMAKE_C_FLAGS=-fPIC
			-DCMAKE_BUILD_TYPE=Release
			-DGLFW_BUILD_EXAMPLES=OFF
			-DGLFW_BUILD_TESTS=OFF
			-DGLFW_BUILD_DOCS=OFF
			-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>)
		list(APPEND THIRD_PARTY_TARGETS glfw)
		list(APPEND GPUIP_LIBRARIES ${LIB_PREFIX}glfw3${LIB_SUFFIX})
		if(UNIX)
		  list(APPEND GPUIP_LIBRARIES X11 Xrandr Xxf86vm Xcursor)
		endif()
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
		add_definitions(-DGLEW_STATIC)
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
		set(BOOST_INSTALL_ARGS --with-python
			link=static variant=release --prefix=${THIRD_PARTY_DIR})
		if(WIN32)
			set(BOOST_CONFIGURE_COMMAND bootstrap.bat)
			set(BOOST_BUILD_COMMAND b2 headers)
			set(BOOST_INSTALL_COMMAND b2 install ${BOOST_INSTALL_ARGS})
			string(REPLACE "." "_" BOOST_VERSION_UNDERSCORE ${BOOST_VERSION})
			include_directories(${THIRD_PARTY_DIR}/include/boost-${BOOST_VERSION_UNDERSCORE})
			#add_definitions(-DBOOST_ALL_NO_LIB )
			add_definitions(-DBOOST_PYTHON_STATIC_LIB)
			if(MSVC8)
				set(BOOST_TOOLSET vc80)
			elseif(MSVC6)
				set(BOOST_TOOLSET vc60)
			elseif(MSVC10)
				set(BOOST_TOOLSET vc100)
			elseif (MSVC12)
				set(BOOST_TOOLSET vc120)
			endif()
			#list(APPEND GPUIP_PYTHON_LIBRARIES
			#		libboost_python-${BOOST_TOOLSET}-mt-${BOOST_VERSION_UNDERSCORE}.lib)
		else()
			set(BOOST_CONFIGURE_COMMAND ./bootstrap.sh)
			set(BOOST_BUILD_COMMAND ./b2 headers)
			set(BOOST_INSTALL_COMMAND ./b2 install ${BOOST_INSTALL_ARGS})
			list(APPEND GPUIP_PYTHON_LIBRARIES ${LIB_PREFIX}boost_python${LIB_SUFFIX})
		endif()
		ExternalProject_Add(boost_python
			GIT_REPOSITORY ${GIT_REPO_BOOST}
			GIT_TAG boost-${BOOST_VERSION}.0
			#GIT_SUBMODULES ${BOOST_GIT_SUBMODULES}
			BUILD_IN_SOURCE 1
			CONFIGURE_COMMAND ${BOOST_CONFIGURE_COMMAND}
			BUILD_COMMAND ${BOOST_BUILD_COMMAND}
			INSTALL_COMMAND ${BOOST_INSTALL_COMMAND})
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
			set(BOOST_STATIC_LIBS_OPTIONS
				-DBOOST_ROOT=${THIRD_PARTY_DIR}
				-DBoost_USE_STATIC_LIBS=ON
				-DBoost_USE_STATIC_RUNTIME=OFF
				-DBoost_USE_MULTITHREADED=ON)
		endif()
		ExternalProject_Add(boost_numpy
			${BOOST_NUMPY_DEPENDS}
			INSTALL_DIR ${THIRD_PARTY_DIR}
			GIT_REPOSITORY ${GIT_REPO_BOOST_NUMPY}
			CMAKE_ARGS
			-DCMAKE_BUILD_TYPE=Release
			-DLIBRARY_TYPE=STATIC
			${BOOST_STATIC_LIBS_OPTIONS}	
			-DBUILD_TESTS=OFF
			-DBUILD_EXAMPLES=OFF
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
		list(APPEND GPUIP_PYTHON_LIBRARIES ${LIB_PREFIX}zlibstatic${LIB_SUFFIX})
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
		  set(CIMG_INSTALL_TARGET INSTALL)
		else()
		  set(CIMG_INSTALL_TARGEt install)
		endif()
		find_program(cmake_bin NAMES cmake)
		ExternalProject_Add(cimg
			GIT_REPOSITORY ${GIT_REPO_CIMG}
			BUILD_IN_SOURCE 1
			PREFIX ${CMAKE_BINARY_DIR}/cimg
			CONFIGURE_COMMAND "" 
			BUILD_COMMAND ${cmake_bin} . 
			-DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_DIR} -Wno-dev)
		# Not too happy about this extra step. Much simpler would just be empty
		# Configure/Build commands and a 'cp CImg.h ${THIRD_PARTY_DIR}/include'
		# as install command. Worked fine in unix/mac but could'nt get the 'copy' 
		# command in windows to work with the args list, hence this 'hack'
		ExternalProject_Add_Step(cimg modify_CMakeLists
			COMMAND echo "install(FILES CImg.h DESTINATION include)" > CMakeLists.txt
			DEPENDEES configure
			DEPENDERS build
			WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/cimg/src/cimg)
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
		if(WIN32)
			list(APPEND GPUIP_PYTHON_LIBRARIES libpng15_static${LIB_SUFFIX})
		else()
			list(APPEND GPUIP_PYTHON_LIBRARIES ${LIB_PREFIX}png${LIB_SUFFIX})
		endif()
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
		list(APPEND GPUIP_TEST_LIBRARIES ${OPENEXR_LIBRARIES})
	elseif(BUILD_THIRD_PARTY_LIBS)
		message(STATUS "${Cyan}Adding third party target openexr${ColorReset}")
		ExternalProject_Add(openexr
			INSTALL_DIR ${THIRD_PARTY_DIR}
			GIT_REPOSITORY ${GIT_REPO_OPENEXR}
			#GIT_TAG v${OPENEXR_VERSION}.0 TODO: Once offical tag builds w/ cmake
			${ZLIB_DEPENDS}
			CMAKE_ARGS
			-DCMAKE_BUILD_TYPE=Release
			-DZLIB_ROOT=${THIRD_PARTY_DIR}
			-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
			-DBUILD_SHARED_LIBS=OFF
			-DBUILD_BIN=OFF
			-DBUILD_TESTS=OFF)
		list(APPEND GPUIP_PYTHON_LIBRARIES ${LIB_PREFIX}Half${LIB_SUFFIX})
		list(APPEND GPUIP_TEST_LIBRARIES ${LIB_PREFIX}Half${LIB_SUFFIX})  
		set(OPENEXR_LIBS Iex IexMath IlmImf IlmThread Imath)
		string(REPLACE "." "_" OPENEXR_VERSION_UNDERSCORE ${OPENEXR_VERSION})
		set(VER -${OPENEXR_VERSION_UNDERSCORE})
		foreach(_L ${OPENEXR_LIBS})
			list(APPEND GPUIP_PYTHON_LIBRARIES ${LIB_PREFIX}${_L}${VER}${LIB_SUFFIX})
			list(APPEND GPUIP_TEST_LIBRARIES ${LIB_PREFIX}${_L}${VER}${LIB_SUFFIX})
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
	add_subdirectory(test)
endif()

if(BUILD_DOCS)
	add_subdirectory(doc)
endif()

if(DOWNLOAD_EXAMPLES_IMAGES)
  message(STATUS "${Cyan}Adding examples_images target${ColorReset}")
		ExternalProject_Add(examples_images
		  GIT_REPOSITORY ${GIT_REPO_EXAMPLES_IMAGES}
		  INSTALL_DIR ${GPUIP_ROOT_DIR}
		  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>)
endif()

# uninstall target
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)
add_custom_target(uninstall
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)