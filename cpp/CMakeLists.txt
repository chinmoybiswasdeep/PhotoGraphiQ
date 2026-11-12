# Native PhotoGraphiQ engines: one pybind11 extension, photographiq._native.
#
# Components are independent so they can be built and tested separately:
#   cmake -S cpp -B build -DPGQ_COMPONENTS="gaussian;fock;sampling"
# Eigen 3.4.0 (MPL-2.0) is found on the system or fetched at a pinned hash.
# Offline builds: -DFETCHCONTENT_SOURCE_DIR_EIGEN=/path/to/eigen-3.4.0
cmake_minimum_required(VERSION 3.24)
project(photographiq_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PGQ_COMPONENTS "gaussian;fock;sampling" CACHE STRING "Native components to build")
option(PGQ_BUILD_CPP_TESTS "Build the C++ unit-test executable" OFF)
option(PGQ_USE_OPENMP "Parallelize kernels with OpenMP when available" ON)

find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT TARGET Eigen3::Eigen)
  include(FetchContent)
  # SOURCE_SUBDIR points nowhere so Eigen's own CMake project (tests, docs) is not configured.
  FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
    SOURCE_SUBDIR do-not-configure
  )
  FetchContent_MakeAvailable(eigen)
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  target_include_directories(Eigen3::Eigen INTERFACE "${eigen_SOURCE_DIR}")
endif()

set(PGQ_CORE_SOURCES "")
set(PGQ_BINDING_SOURCES "")
set(PGQ_DEFINITIONS "")
foreach(component IN LISTS PGQ_COMPONENTS)
  if(NOT component MATCHES "^(gaussian|fock|sampling)$")
    message(FATAL_ERROR "Unknown native component '${component}'")
  endif()
  file(GLOB component_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/${component}/*.cpp")
  list(APPEND PGQ_CORE_SOURCES ${component_sources})
  list(APPEND PGQ_BINDING_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings/${component}_bindings.cpp")
  string(TOUPPER "${component}" upper)
  list(APPEND PGQ_DEFINITIONS "PGQ_WITH_${upper}=1")
endforeach()
file(GLOB core_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp")
list(APPEND PGQ_CORE_SOURCES ${core_sources})

add_library(pgq_core STATIC ${PGQ_CORE_SOURCES})
target_include_directories(pgq_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(pgq_core PUBLIC Eigen3::Eigen)
# Eigen code paths restricted to MPL-2.0 licensed sources.
target_compile_definitions(pgq_core PUBLIC EIGEN_MPL2_ONLY ${PGQ_DEFINITIONS})
if(MSVC)
  target_compile_options(pgq_core PUBLIC /W4 /permissive- /utf-8 /bigobj /EHsc /Zc:__cplusplus)
else()
  target_compile_options(pgq_core PUBLIC -Wall -Wextra -Wpedantic)
endif()
if(PGQ_USE_OPENMP)
  find_package(OpenMP COMPONENTS CXX)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(pgq_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(pgq_core PUBLIC PGQ_HAVE_OPENMP=1)
  endif()
endif()

pybind11_add_module(_native "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings/module.cpp" ${PGQ_BINDING_SOURCES})
target_link_libraries(_native PRIVATE pgq_core)
target_compile_definitions(_native PRIVATE
  PGQ_COMPILER_ID="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
  PGQ_BUILD_TYPE="$<IF:$<CONFIG:>,${CMAKE_BUILD_TYPE},$<CONFIG>>")
install(TARGETS _native LIBRARY DESTINATION photographiq RUNTIME DESTINATION photographiq)

if(PGQ_BUILD_CPP_TESTS)
  file(GLOB test_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp")
  enable_testing()
  foreach(test_source IN LISTS test_sources)
    get_filename_component(test_name "${test_source}" NAME_WE)
    # Each test file is guarded by the PGQ_WITH_* macros of the components it needs.
    add_executable(${test_name} "${test_source}")
    target_link_libraries(${test_name} PRIVATE pgq_core)
    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()
endif()
