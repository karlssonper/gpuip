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

# Build with OpenCL
if(OpenCL_FOUND AND BUILD_WITH_OPENCL)
  include_directories(${OpenCL_INCLUDE_DIRS})
  add_library(gpuipOpenCL SHARED opencl)
  install(TARGETS gpuipOpenCL DESTINATION lib COMPONENT devel)
  target_link_libraries(gpuipOpenCL ${OpenCL_LIBRARIES})
endif()

# Build with CUDA
if(CUDA_FOUND AND BUILD_WITH_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
  add_library(gpuipCUDA SHARED cuda)
  install(TARGETS gpuipCUDA DESTINATION lib COMPONENT devel)
  target_link_libraries(gpuipCUDA ${CUDA_LIBRARIES} ${CUDA_DRIVER_LIBRARY})
endif()

# Build with GLSL (needs OpenGL)
if(OPENGL_FOUND AND BUILD_WITH_GLSL)
  add_library(gpuipGLSL SHARED glsl)
  target_link_libraries(gpuipGLSL ${GPUIP_OPENGL_LIBRARIES})
  install(TARGETS gpuipGLSL DESTINATION lib COMPONENT devel)
  if (OPENGL_THIRD_PARTY_TARGETS)
	add_dependencies(gpuipGLSL ${OPENGL_THIRD_PARTY_TARGETS})
  endif()
endif()

# Build the gpuip library
if(BUILD_SHARED_LIB)
  set(LIBRARY_TYPE SHARED)
else()
  set(LIBRARY_TYPE STATIC)
endif()
add_definitions(-DGPUIP_VERSION=${GPUIP_VERSION})
add_library(gpuip ${LIBRARY_TYPE} gpuip implinterface)
target_link_libraries(gpuip ${GPUIP_LIBRARIES} dl)
install(TARGETS gpuip DESTINATION lib COMPONENT devel)
install(FILES gpuip.h DESTINATION include COMPONENT devel)
if (THIRD_PARTY_TARGETS)
  add_dependencies(gpuip ${THIRD_PARTY_TARGETS})
endif()

# Build python bindings (using boost python)
if(BUILD_PYTHON_BINDINGS)
  add_library(_pygpuip SHARED python.cpp io_wrapper.cpp)
  target_link_libraries(_pygpuip gpuip ${GPUIP_PYTHON_LIBRARIES})

  # Rename python shared lib from libpyGpuip.{so,lib} to pyGpuip.{so,pyd}
  set_target_properties(_pygpuip PROPERTIES PREFIX "")
  if(WIN32)
    set_target_properties(_pygpuip PROPERTIES SUFFIX ".pyd")
  else()
    set_target_properties(_pygpuip PROPERTIES SUFFIX ".so")
  endif()

  # Install python bindings in the python site-packages
  execute_process(COMMAND
    ${PYTHON_EXECUTABLE} -c
    "from distutils.sysconfig import get_python_lib; print get_python_lib()"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
  install(TARGETS _pygpuip DESTINATION ${PYTHON_SITE_PACKAGES} COMPONENT python)
  install(FILES pygpuip.py DESTINATION ${PYTHON_SITE_PACKAGES} COMPONENT python)
endif()



