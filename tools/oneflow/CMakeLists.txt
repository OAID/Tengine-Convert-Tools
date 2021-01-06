find_package(Protobuf REQUIRED)

FILE(GLOB_RECURSE ONEFLOW_SERIALIZER_PROTOS "*.proto")
# It is a undocumented variable
set(PROTOBUF_GENERATE_CPP_APPEND_PATH FALSE)
protobuf_generate_cpp(ONEFLOW_PROTO_SRCS ONEFLOW_PROTO_HDRS ${ONEFLOW_SERIALIZER_PROTOS})

FILE(GLOB_RECURSE ONEFLOW_SERIALIZER_SRCS "*.cpp")
add_library(oneflow2tengine ${ONEFLOW_SERIALIZER_SRCS} ${ONEFLOW_PROTO_SRCS})
if (${CMAKE_VERSION} VERSION_LESS "3.9.0")
    target_link_libraries(oneflow2tengine ${CMAKE_PROJECT_NAME} ${Protobuf_LIBRARIES})
    target_include_directories(oneflow2tengine PUBLIC ${Protobuf_INCLUDE_DIRS})
else()
    target_link_libraries(oneflow2tengine ${CMAKE_PROJECT_NAME} protobuf::libprotobuf)
endif()
target_include_directories(oneflow2tengine PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
