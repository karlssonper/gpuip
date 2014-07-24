# Locate the glfw library
# This module defines the following variables:
# GLFW_LIBRARY, the name of the library;
# GLFW_INCLUDE_DIR, where to find glfw include files.
# GLFW_FOUND, true if both the GLFW_LIBRARY and GLFW_INCLUDE_DIR have been found.
#
# To help locate the library and include file, you could define an environment variable called
# GLFW_ROOT which points to the root of the glfw library installation. This is pretty useful
# on a Windows platform.
#
#
# Usage example to compile an "executable" target to the glfw library:
#
# FIND_PACKAGE (glfw REQUIRED)
# INCLUDE_DIRECTORIES (${GLFW_INCLUDE_DIR})
# ADD_EXECUTABLE (executable ${EXECUTABLE_SRCS})
# TARGET_LINK_LIBRARIES (executable ${GLFW_LIBRARY})
#
# TODO:
# Allow the user to select to link to a shared library or to a static library.

#Search for the include file...
FIND_PATH(GLFW_INCLUDE_DIRS GLFW/glfw3.h DOC "Path to GLFW include directory."
  HINTS
  $ENV{GLFW_ROOT}
  PATH_SUFFIX include #For finding the include file under the root of the glfw expanded archive, typically on Windows.
  PATHS
  /usr/include/
  /usr/local/include/
  # By default headers are under GL subfolder
  /usr/include/GL
  /usr/local/include/GL
  "C:/Program Files (x86)/GLFW/include"
  ${GLFW_ROOT_DIR}/include/ # added by ptr
 
)

FIND_LIBRARY(GLFW_LIBRARIES DOC "Absolute path to GLFW library."
  NAMES glfw glfw3
  HINTS
  $ENV{GLFW_ROOT}
  PATH_SUFFIXES lib/win32 #For finding the library file under the root of the glfw expanded archive, typically on Windows.
  PATHS
   "C:/Program Files (x86)/GLFW/lib"
  /usr/local/lib
  /usr/lib
  ${GLFW_ROOT_DIR}/lib-msvc100/release # added by ptr
)

if (APPLE)
  set(GLFW_cocoa "-framework Cocoa" CACHE STRING "Cocoa framework for OSX")
  set(GLFW_corevideo "-framework CoreVideo" CACHE STRING "CoreVideo framework for OSX")
  set(GLFW_iokit "-framework IOKit" CACHE STRING "IOKit framework for OSX")
  set(GLFW_LIBRARIES ${GLFW_LIBRARIES} ${GLFW_cocoa} ${GLFW_corevideo} ${GLFW_iokit})
endif(APPLE)

SET(GLFW_FOUND 0)
IF(GLFW_LIBRARY AND GLFW_INCLUDE_DIR)
  SET(GLFW_FOUND 1)
  message(STATUS "GLFW found!")
ENDIF(GLFW_LIBRARY AND GLFW_INCLUDE_DIR)
