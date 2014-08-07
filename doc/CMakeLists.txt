find_package(Doxygen REQUIRED)
add_custom_target(doc ALL ${DOXYGEN_EXECUTABLE}
                  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/doc
                  COMMENT "Generating documentation" VERBATIM)