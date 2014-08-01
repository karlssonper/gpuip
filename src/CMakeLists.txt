cmake_minimum_required(VERSION 2.8)

project(gpuip_src)

if (NOT WIN32)
  add_definitions(-Wall)
  add_definitions(-fPIC)
endif()

# Common variables for compling the library
set(SOURCE gpuip)
set(LIBRARIES)

# Build with OpenCL
if(OpenCL)
  message(STATUS "Generating build with OpenCL...")
  add_definitions(-D_GPUIP_OPENCL)
  find_package(OpenCL REQUIRED)
  include_directories(${OPENCL_INCLUDE_DIRS})
  message(${OPENCL_LIBRARIES})
  set(LIBRARIES ${LIBRARIES} ${OPENCL_LIBRARIES})
  set(SOURCE ${SOURCE} opencl)  
else(OpenCL)
  message(STATUS "OpenCL disabled .")
endif(OpenCL)

# Build with CUDA
if(CUDA)
  message(STATUS "Generating build with CUDA...")
  add_definitions(-D_GPUIP_CUDA)
  find_package(CUDA REQUIRED)
  find_package(CUDADriver REQUIRED)
  include_directories(${CUDA_INCLUDE_DIRS})

  set(LIBRARIES ${LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_DRIVER_LIBRARY})
  set(SOURCE ${SOURCE} cuda)  
else(CUDA)
  message(STATUS "CUDA disabled .")
endif(CUDA)

# Build with GLSL (needs OpenGL)
if(GLSL)
  message(STATUS "Generating build with GLSL...")
  add_definitions(-D_GPUIP_GLSL)
  find_package(OpenGL REQUIRED)   
  
  # Easier to set these manually in windows
  if (WIN32)
	set(GLEW_INCLUDE_DIRS $ENV{GLEW_INCLUDE_DIR})
	set(GLEW_LIBRARY $ENV{GLEW_LIBRARY_DIR}/glew32.lib)
  else (WIN32)
	find_package(GLEW REQUIRED)
  endif(WIN32)
  find_package(GLFW REQUIRED)
  include_directories(${OPENGL_INCLUDE_DIRS} 
                      ${GLEW_INCLUDE_DIRS} 
                      ${GLFW_INCLUDE_DIRS})  
  set(SOURCE ${SOURCE} glsl)
  if (APPLE)
	set(SOURCE ${SOURCE} glcontext.m)
  endif (APPLE)
  set(LIBRARIES ${LIBRARIES} ${OPENGL_LIBRARIES} 
	${GLFW_LIBRARIES} ${GLEW_LIBRARY})
else(GLSL)
  message(STATUS "GLSL disabled .")
endif(GLSL)

# Build the gpuip library
add_library(gpuip_static ${SOURCE})
add_library(gpuip SHARED ${SOURCE})
target_link_libraries(gpuip_static ${LIBRARIES})
target_link_libraries(gpuip ${LIBRARIES})
install(TARGETS gpuip DESTINATION lib)

# Build python bindings (using boost python)
if(PYTHON)
  message(STATUS "Generating build with python bindings...")
  add_definitions(-D_GPUIP_PYTHON_BINDINGS)
  find_package(Boost REQUIRED COMPONENTS python)
  find_package(BoostNumpy REQUIRED)
  find_package(PythonLibs REQUIRED)
  find_package(PNG)
  find_package(TIFF)
  find_package(JPEG)
  find_package(OpenEXR)

  include_directories(
	${PYTHON_INCLUDE_DIR} 
	${Boost_INCLUDE_DIRS}
	${Boost_NUMPY_INCLUDE_DIRS})

  set(PY_LIBS 
	gpuip_static 
	${PYTHON_LIBRARIES} 
	${Boost_PYTHON_LIBRARY} 
	${Boost_NUMPY_LIBRARY})

  if(PNG_FOUND)
	add_definitions(-Dcimg_use_png)
	include_directories(${PNG_INCLUDE_DIRS})
	set(PY_LIBS ${PY_LIBS} ${PNG_LIBRARIES})
  endif(PNG_FOUND)

  if(JPEG_FOUND)
	add_definitions(-Dcimg_use_jpeg)
	include_directories(${JPEG_INCLUDE_DIR})
	set(PY_LIBS ${PY_LIBS} ${JPEG_LIBRARIES})
  endif(JPEG_FOUND)

  if(TIFF_FOUND)
	add_definitions(-Dcimg_use_tiff)
	include_directories(${TIFF_INCLUDE_DIR})
	set(PY_LIBS ${PY_LIBS} ${TIFF_LIBRARIES})
  endif(TIFF_FOUND)

  if(OPENEXR_FOUND)
	add_definitions(-Dcimg_use_exr)
	include_directories(${OPENEXR_INCLUDE_PATHS})
	set(PY_LIBS ${PY_LIBS} ${OPENEXR_LIBRARIES})
  endif(OPENEXR_FOUND)
  
  add_library(pygpuip SHARED python.cpp cimg_numpy_wrapper)
  target_link_libraries(pygpuip ${PY_LIBS})
  
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