add_library(lib_dep STATIC block.cpp grid.cpp nn_blocks.cpp cuda_kernels.cpp)
set_target_properties (lib_dep
  PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  PREFIX                    "lib"
  OUTPUT_NAME               "dep"
  LIBRARY_OUTPUT_DIRECTORY  "${CMAKE_BINARY_DIR}/lib"
)

