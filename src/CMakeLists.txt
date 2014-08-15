cmake_minimum_required(VERSION 2.8)

project(gpuip_src)

if (NOT WIN32)
  add_definitions(-Wall)
  add_definitions(-fPIC)
  add_definitions(-O3)
endif()

# Common variables for compling the library
set(SOURCE gpuip)

# Build with OpenCL
if(OPENCL)
  message(STATUS "Generating build with OpenCL...")
  add_definitions(-D_GPUIP_OPENCL)
  set(SOURCE ${SOURCE} opencl)  
else(OPENCL)
  message(STATUS "OpenCL disabled .")
endif(OPENCL)

# Build with CUDA
if(CUDA)
  message(STATUS "Generating build with CUDA...")
  add_definitions(-D_GPUIP_CUDA)
  if(CUDA_HELPER_MATH_FOUND)
	add_definitions(-DCUDA_HELPER_DIR="\\"${CUDA_HELPER_MATH_DIR}\\"")
  else(CUDA_HELPER_MATH_FOUND)
	message(STATUS "CUDA Helper math not included")  
  endif(CUDA_HELPER_MATH_FOUND)
  set(SOURCE ${SOURCE} cuda)  
else(CUDA)
  message(STATUS "CUDA disabled .")
endif(CUDA)

# Build with GLSL (needs OpenGL)
if(GLSL)
  message(STATUS "Generating build with GLSL...")
  add_definitions(-D_GPUIP_GLSL)
  set(SOURCE ${SOURCE} glsl)
  if (APPLE)
	set(SOURCE ${SOURCE} glcontext.m)
  endif (APPLE)
else(GLSL)
  message(STATUS "GLSL disabled .")
endif(GLSL)

# Build the gpuip library
if (BUILD_SHARED_LIB)
  add_library(gpuip SHARED ${SOURCE})
else (BUILD_SHARED_LIB)
  add_library(gpuip ${SOURCE})
endif (BUILD_SHARED_LIB)
target_link_libraries(gpuip ${GPUIP_LIBRARIES})
install(TARGETS gpuip DESTINATION lib)
if (BUILD_THIRD_PARTY_LIBS AND MISSING_THIRD_PARTY)
  add_dependencies(gpuip third_party)
endif (BUILD_THIRD_PARTY_LIBS AND MISSING_THIRD_PARTY)

# Build python bindings (using boost python)
if(PYTHON)
  message(STATUS "Generating build with python bindings...")
  add_library(pygpuip SHARED python.cpp io_wrapper.cpp)
  target_link_libraries(pygpuip ${GPUIP_PYTHON_LIBRARIES})
  
  # Rename python shared lib from libpyGpuip.{so,lib} to pyGpuip.{so,pyd}
  set_target_properties( pygpuip PROPERTIES PREFIX "")
  if (WIN32)
	set_target_properties( pygpuip PROPERTIES SUFFIX ".pyd")
  else (WIN32)
	set_target_properties( pygpuip PROPERTIES SUFFIX ".so")
  endif(WIN32)

  # Install python bindings in the python site-packages
  execute_process(COMMAND 
	python -c "from distutils.sysconfig import get_python_lib; print get_python_lib()" 
	OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
  install(TARGETS pygpuip DESTINATION ${PYTHON_SITE_PACKAGES})
else(PYTHON)
  message(STATUS "Python bindings disabled.")
endif(PYTHON)