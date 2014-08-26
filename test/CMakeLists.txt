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

project(gpuip_test)

include_directories(${GPUIP_ROOT_DIR}/src)

# Add C++ test
add_executable(test_cpp test)
target_link_libraries(test_cpp gpuip)
add_test(NAME test_cpp COMMAND test_cpp)

set(GPUIP_TEST_IMAGE ${GPUIP_ROOT_DIR}/examples/images/GoldenGate.exr)
if(GPUIP_TEST_LIBRARIES AND 
	(EXISTS ${GPUIP_TEST_IMAGE} OR DOWNLOAD_EXAMPLES_IMAGES))
  find_package(OpenMP)
  if (OPENMP_FOUND)
	add_definitions(-D_GPUIP_TEST_WITH_OPENMP)
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
  add_executable(test_performance performance)
  target_link_libraries(test_performance gpuip ${GPUIP_TEST_LIBRARIES})
  set(GPUIP_TEST_KERNEL_DIR ${GPUIP_ROOT_DIR}/examples/kernels)
  add_test(NAME test_performance 
	COMMAND test_performance ${GPUIP_TEST_IMAGE} ${GPUIP_TEST_KERNEL_DIR})
endif()

# Add python test
if(BUILD_PYTHON_BINDINGS)
  configure_file(test.py ../src/test.py COPYONLY)
  add_test(NAME test_py COMMAND ${PYTHON_EXECUTABLE} ../src/test.py)
endif()
