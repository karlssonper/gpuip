find_package(Doxygen REQUIRED)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
  ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
add_custom_target(doc ALL ${DOXYGEN_EXECUTABLE}
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  COMMENT "Generating documentation" VERBATIM)