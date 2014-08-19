project(gpuip_test)

include_directories(${GPUIP_ROOT_DIR}/src)

# Add C++ test
add_executable(test_cpp test)
target_link_libraries(test_cpp gpuip)
add_test(NAME test_cpp COMMAND test_cpp)

if(GPUIP_TEST_LIBRARIES)
  find_package(OpenMP)
  if (OPENMP_FOUND)
	add_definitions(-D_GPUIP_TEST_WITH_OPENMP)
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
  add_executable(test_speed speed)
  target_link_libraries(test_speed gpuip ${GPUIP_TEST_LIBRARIES})
  set(GPUIP_TEST_IMAGE ${GPUIP_ROOT_DIR}/examples/images/GoldenGate.exr)
  set(GPUIP_TEST_KERNEL_DIR ${GPUIP_ROOT_DIR}/examples/kernels)
  add_test(NAME test_speed 
	COMMAND test_speed ${GPUIP_TEST_IMAGE} ${GPUIP_TEST_KERNEL_DIR})
endif()

# Add python test
if(BUILD_PYTHON_BINDINGS)
  find_package(PythonInterp REQUIRED)
  configure_file(test.py src/test.py COPYONLY)
  add_test(NAME test_py COMMAND ${PYTHON_EXECUTABLE} src/test.py)
endif()
