set(executable grid)
add_executable(${executable} app.cpp)
target_link_libraries(${executable} lib_dep lbm)
#####set_target_properties(${executable} PROPERTIES CUDA_ARCHITECTURES "35;50;72")

set_property(TARGET "${executable}"
  PROPERTY
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

