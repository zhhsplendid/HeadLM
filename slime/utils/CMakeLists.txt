file(GLOB CPP_SOURCES "*.cpp")

add_library(utils SHARED ${CPP_SOURCES})
target_link_libraries(utils PUBLIC info_struct)
