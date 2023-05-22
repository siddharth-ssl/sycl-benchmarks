add_library(lbm STATIC D3Q27SC.cpp)
set_target_properties (lbm 
  PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  PREFIX                    "lib"
  OUTPUT_NAME               "lbm"
  LIBRARY_OUTPUT_DIRECTORY  "${CMAKE_BINARY_DIR}/lib"
)

