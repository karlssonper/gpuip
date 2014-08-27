# The MIT License (MIT)
# 
# Copyright (c) 2014 Per Karlsson
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

project(gpuip_src)

if(NOT WIN32)
  add_definitions(-Wall)
  add_definitions(-fPIC)
  add_definitions(-O3)
endif()

# Common variables for compling the library
set(SOURCE gpuip)

# Build with OpenCL
if(OpenCL_FOUND AND BUILD_WITH_OPENCL)
  add_definitions(-D_GPUIP_OPENCL)
  set(SOURCE ${SOURCE} opencl)
endif()

# Build with CUDA
if(CUDA_FOUND AND BUILD_WITH_CUDA)
  add_definitions(-D_GPUIP_CUDA)
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
add_definitions(-DGPUIP_VERSION=${GPUIP_VERSION})
add_library(gpuip ${LIBRARY_TYPE} ${SOURCE})
target_link_libraries(gpuip ${GPUIP_LIBRARIES})
install(TARGETS gpuip DESTINATION lib COMPONENT devel)
install(FILES gpuip.h DESTINATION include COMPONENT devel)
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
    ${PYTHON_EXECUTABLE} -c
    "from distutils.sysconfig import get_python_lib; print get_python_lib()"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
  install(TARGETS pygpuip DESTINATION ${PYTHON_SITE_PACKAGES} COMPONENT python)
endif()



