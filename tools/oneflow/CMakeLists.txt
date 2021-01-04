find_package(Protobuf REQUIRED)

FILE(GLOB_RECURSE ONEFLOW_SERIALIZER_PROTOS "*.proto")
# It is a undocumented variable
set(PROTOBUF_GENERATE_CPP_APPEND_PATH FALSE)
protobuf_generate_cpp(ONEFLOW_PROTO_SRCS ONEFLOW_PROTO_HDRS ${ONEFLOW_SERIALIZER_PROTOS})

FILE(GLOB_RECURSE ONEFLOW_SERIALIZER_SRCS "*.cpp")
add_library(oneflow2tengine ${ONEFLOW_SERIALIZER_SRCS} ${ONEFLOW_PROTO_SRCS})
target_link_libraries(oneflow2tengine ${CMAKE_PROJECT_NAME} protobuf::libprotobuf)
target_include_directories(oneflow2tengine PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
