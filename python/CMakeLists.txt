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

project(gpuip_python)

set(PYTHON_ICONS_FILE ${GPUIP_ROOT_DIR}/python/icons.py)
add_custom_target(generateIcons ALL 
  ${PYTHON_EXECUTABLE} generateIcons.py ${PYTHON_ICONS_FILE}
  WORKING_DIRECTORY ${GPUIP_ROOT_DIR}/python/icons)

file(GLOB PYTHON_FILES "*.py")
install(FILES ${PYTHON_FILES} ${PYTHON_ICONS_FILE} DESTINATION gpuip
  COMPONENT bin)
install(FILES gpuip.py DESTINATION gpuip
  PERMISSIONS OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE
  OWNER_READ GROUP_READ WORLD_READ COMPONENT bin)

if(NOT WIN32)
  # Create symlink in CMAKE_INSTALL_PREFIX/bin
  set(GPUIP_PY ${CMAKE_INSTALL_PREFIX}/gpuip/gpuip.py)
  add_custom_target(symlink ALL 
	ln -sf ${GPUIP_PY} gpuip
	WORKING_DIRECTORY ${GPUIP_ROOT_DIR}/python)
  install(FILES gpuip DESTINATION bin COMPONENT bin)
endif()






