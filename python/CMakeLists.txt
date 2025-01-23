message(STATUS "================  PYTHON BINDINGS ENABLED. BUILDING BINDINGS  ======================")

# Fetch pybind to build the python bindings
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.0
  FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(pybind11)

# TODO: See if this is actually needed
set_target_properties(loam PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

# Declare the pybind target
file(GLOB_RECURSE pybind_srcs "*.cpp")
pybind11_add_module(loam_python ${pybind_srcs})
target_link_libraries(loam_python PUBLIC loam)
install(TARGETS loam_python DESTINATION loam)