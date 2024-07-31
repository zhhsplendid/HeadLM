
include(ExternalProject)

enable_testing()

set(GTEST_PREFIX_DIR ${THIRD_PARTY_PATH}/gtest)
set(GTEST_INSTALL_DIR ${THIRD_PARTY_PATH}/install/gtest)
set(GTEST_INCLUDE_DIR
    "${GTEST_INSTALL_DIR}/include"
    CACHE PATH "gtest include directory." FORCE)
set(SOURCE_DIR ${HEADLM_SOURCE_DIR}/third_party/gtest)
set(GTEST_TAG v1.11.x)
set(GTEST_SOURCE_DIR ${THIRD_PARTY_PATH}/gtest/src/extern_gtest)
include_directories(SYSTEM ${GTEST_INCLUDE_DIR})
include_directories(SYSTEM ${GTEST_INCLUDE_DIR}/gtest)
include_directories(SYSTEM ${GTEST_INCLUDE_DIR}/gtest/internal)

set(GTEST_LIBRARIES
      "${GTEST_INSTALL_DIR}/lib/libgtest.a"
      CACHE FILEPATH "gtest libraries." FORCE)
set(GTEST_MAIN_LIBRARIES
    "${GTEST_INSTALL_DIR}/lib/libgtest_main.a"
    CACHE FILEPATH "gtest main libraries." FORCE)
set(GMOCK_LIBRARIES
    "${GTEST_INSTALL_DIR}/lib/libgmock.a"
    CACHE FILEPATH "gmock libraries." FORCE)

# The Googletest has treated warning as error, but it contains bug
# Set compiler flags with -w to disable warnings so it can compile
set(GTEST_CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w")
set(GTEST_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")

ExternalProject_Add(
    extern_gtest
    ${EXTERNAL_PROJECT_LOG_ARGS}
    SOURCE_DIR ${SOURCE_DIR}
    DEPENDS ${GTEST_DEPENDS}
    PREFIX ${GTEST_PREFIX_DIR}
    UPDATE_COMMAND ""
    PATCH_COMMAND ${GTEST_PATCH_COMMAND}
    CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
               -DCMAKE_CXX_FLAGS=${GTEST_CMAKE_CXX_FLAGS}
               -DCMAKE_CXX_FLAGS_RELEASE=${GTEST_CMAKE_CXX_FLAGS}
               -DCMAKE_CXX_FLAGS_DEBUG=${GTEST_CMAKE_CXX_FLAGS}
               -DCMAKE_C_FLAGS=${GTEST_CMAKE_C_FLAGS}
               -DCMAKE_C_FLAGS_DEBUG=${GTEST_CMAKE_C_FLAGS}
               -DCMAKE_C_FLAGS_RELEASE=${GTEST_CMAKE_C_FLAGS}
               -DCMAKE_INSTALL_PREFIX=${GTEST_INSTALL_DIR}
               -DCMAKE_POSITION_INDEPENDENT_CODE=ON
               -DBUILD_GMOCK=ON
               -Dgtest_disable_pthreads=OFF
               -Dgtest_force_shared_crt=OFF
               -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
               ${EXTERNAL_OPTIONAL_ARGS}
    CMAKE_CACHE_ARGS
      -DCMAKE_INSTALL_PREFIX:PATH=${GTEST_INSTALL_DIR}
      -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
      -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
    CMAKE_GENERATOR "Unix Makefiles"
    BUILD_BYPRODUCTS ${GTEST_LIBRARIES}
    BUILD_BYPRODUCTS ${GTEST_MAIN_LIBRARIES}
    BUILD_BYPRODUCTS ${GMOCK_LIBRARIES})

message(STATUS "GTEST_LIBRARIES=${GTEST_LIBRARIES}")
message(STATUS "GTEST_MAIN_LIBRARIES=${GTEST_MAIN_LIBRARIES}")

add_library(gtest STATIC IMPORTED GLOBAL)
set_property(TARGET gtest PROPERTY IMPORTED_LOCATION ${GTEST_LIBRARIES})
add_dependencies(gtest extern_gtest)

add_library(gtest_main STATIC IMPORTED GLOBAL)
set_property(TARGET gtest_main PROPERTY IMPORTED_LOCATION
                                        ${GTEST_MAIN_LIBRARIES})
add_dependencies(gtest_main extern_gtest)

add_library(gmock STATIC IMPORTED GLOBAL)
set_property(TARGET gmock PROPERTY IMPORTED_LOCATION ${GMOCK_LIBRARIES})
add_dependencies(gmock extern_gtest)

