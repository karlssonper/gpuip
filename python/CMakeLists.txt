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






