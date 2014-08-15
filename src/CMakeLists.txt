cmake_minimum_required(VERSION 2.8)

project(gpuip_src)

if(NOT WIN32)
  add_definitions(-Wall)
  add_definitions(-fPIC)
  add_definitions(-O3)
endif()

# Common variables for compling the library
set(SOURCE gpuip)

# Build with OpenCL
if(OPENCL_FOUND AND BUILD_WITH_OPENCL)
  add_definitions(-D_GPUIP_OPENCL)
  set(SOURCE ${SOURCE} opencl)
endif()

# Build with CUDA
if(CUDA_FOUND AND BUILD_WITH_CUDA)
  add_definitions(-D_GPUIP_CUDA)
  if(CUDA_HELPER_MATH_FOUND)
    add_definitions(-DCUDA_HELPER_DIR="\\"${CUDA_HELPER_MATH_DIR}\\"")
  else(CUDA_HELPER_MATH_FOUND)
    message(STATUS "CUDA Helper math not included")
  endif(CUDA_HELPER_MATH_FOUND)
  set(SOURCE ${SOURCE} cuda)
endif()

# Build with GLSL (needs OpenGL)
if(OPENGL_FOUND AND BUILD_WITH_GLSL)
  add_definitions(-D_GPUIP_GLSL)
  set(SOURCE ${SOURCE} glsl)
  if(APPLE)
    set(SOURCE ${SOURCE} glcontext.m)
  endif()
endif()

# Build the gpuip library
if(BUILD_SHARED_LIB)
  set(LIBRARY_TYPE SHARED)
else()
  set(LIBRARY_TYPE STATIC)
endif()
add_library(gpuip ${LIBRARY_TYPE} ${SOURCE})
target_link_libraries(gpuip ${GPUIP_LIBRARIES})
install(TARGETS gpuip DESTINATION lib)
if (THIRD_PARTY_TARGETS)
  add_dependencies(gpuip ${THIRD_PARTY_TARGETS})
endif()

# Build python bindings (using boost python)
if(BUILD_PYTHON_BINDINGS)
  add_library(pygpuip SHARED python.cpp io_wrapper.cpp)
  target_link_libraries(pygpuip gpuip ${GPUIP_PYTHON_LIBRARIES})

  # Rename python shared lib from libpyGpuip.{so,lib} to pyGpuip.{so,pyd}
  set_target_properties(pygpuip PROPERTIES PREFIX "")
  if(WIN32)
    set_target_properties(pygpuip PROPERTIES SUFFIX ".pyd")
  else()
    set_target_properties(pygpuip PROPERTIES SUFFIX ".so")
  endif()

  # Install python bindings in the python site-packages
  execute_process(COMMAND
    python -c
    "from distutils.sysconfig import get_python_lib; print get_python_lib()"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
  install(TARGETS pygpuip DESTINATION ${PYTHON_SITE_PACKAGES})
endif()



