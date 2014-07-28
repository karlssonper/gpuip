cmake_minimum_required(VERSION 2.8)

project(gpuip_test)

# Enable unit testing
enable_testing()

# Add C++ test
add_executable(test_cpp test)
target_link_libraries(test_cpp gpuip_static)
add_test(NAME test_cpp COMMAND test_cpp)

# Add python test
if (PYTHON)
  find_package(PythonInterp REQUIRED)
  configure_file(test.py test.py COPYONLY)
  add_test(NAME test_py COMMAND ${PYTHON_EXECUTABLE} test.py)
endif(PYTHON)