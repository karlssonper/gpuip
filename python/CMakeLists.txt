 file(GLOB PYTHON_FILES "*.py")
 
 install(FILES ${PYTHON_FILES} DESTINATION gpuip)
 install(FILES gpuip.py DESTINATION gpuip
   PERMISSIONS OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE 
   OWNER_READ GROUP_READ WORLD_READ )

if(NOT WIN32)
  if (NOT EXISTS ${CMAKE_INSTALL_PREFIX}/bin/)
	install(CODE "execute_process(COMMAND mkdir ${CMAKE_INSTALL_PREFIX}/bin)")
  endif()
  # Create symlink in CMAKE_INSTALL_PREFIX/bin
  install(CODE "execute_process(COMMAND ln -sf ${CMAKE_INSTALL_PREFIX}/gpuip/gpuip.py ${CMAKE_INSTALL_PREFIX}/bin/gpuip)")
endif(NOT WIN32)

 

   

