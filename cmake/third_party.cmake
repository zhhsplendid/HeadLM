set(THIRD_PARTY_PATH
    "${CMAKE_BINARY_DIR}/third_party"
    CACHE STRING
          "The path of the directory which downloads & builds third party libraries.")

set(THIRD_PARTY_BUILD_TYPE Release)

include(cmake/external/gtest.cmake)
