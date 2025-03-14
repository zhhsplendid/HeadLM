file(GLOB INFO_STRUCT_SOURCES "*.cpp")

add_library(info_struct SHARED ${INFO_STRUCT_SOURCES})
target_link_libraries(info_struct PUBLIC flatbuffers)