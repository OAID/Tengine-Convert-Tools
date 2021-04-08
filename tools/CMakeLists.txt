CONFIGURE_FILE (config.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/config.hpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/tflite)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/tflite/flatbuffers)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/helper)
include_directories(${CMAKE_CURRENT_BINARY_DIR})


# add protobuf
if(BUILD_CAFFE_SERIALIZER OR BUILD_ONNX_SERIALIZER OR BUILD_TF_SERIALIZER)
    find_package(Protobuf 3.0.0 REQUIRED)
    include_directories(${Protobuf_INCLUDE_DIR})
endif()



# CAFFE SERIALIZER
if(BUILD_CAFFE_SERIALIZER)
    file(GLOB_RECURSE CAFFE_SERIALIZER_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/caffe/*.cpp")

    # the generated caffe.pb.cc

    set(CAFFE_PROTO_SRC        ${CMAKE_CURRENT_BINARY_DIR}/caffe/te_caffe.pb.cc)
    set(CAFFE_PROTO_PATH       ${CMAKE_CURRENT_SOURCE_DIR}/caffe)
    set(CAFFE_PROTO_OUT_PATH   ${CMAKE_CURRENT_BINARY_DIR}/caffe)

    add_custom_command(OUTPUT  ${CAFFE_PROTO_SRC}
                       COMMAND mkdir -p ${CAFFE_PROTO_OUT_PATH}
                       COMMAND ${Protobuf_PROTOC_EXECUTABLE} --cpp_out=${CAFFE_PROTO_OUT_PATH} --proto_path=${CAFFE_PROTO_PATH} ${CAFFE_PROTO_PATH}/te_caffe.proto
                       #COMMAND mv ${CAFFE_PROTO_OUT_PATH}/te_caffe.pb.h ${CAFFE_PROTO_PATH}/../include/
                       DEPENDS ${CAFFE_PROTO_PATH}/te_caffe.proto)

    add_custom_target(CAFFE_SERIALIZER_TARGET DEPENDS ${CAFFE_PROTO_SRC})

    include_directories(${CAFFE_PROTO_OUT_PATH})

    list(APPEND CAFFE_SERIALIZER_SRCS ${CAFFE_PROTO_SRC})
    #message("Caffe serializer src: ${CAFFE_SERIALIZER_SRCS}")
endif()



# ONNX SERIALIZER
if(BUILD_ONNX_SERIALIZER)
    FILE(GLOB_RECURSE ONNX_SERIALIZER_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/onnx/*.cpp")

    # the generated pb.cc

    set(ONNX_PROTO_SRC         ${CMAKE_CURRENT_BINARY_DIR}/onnx/onnx.pb.cc)
    set(ONNX_PROTO_PATH        ${CMAKE_CURRENT_SOURCE_DIR}/onnx)
    set(ONNX_PROTO_OUT_PATH    ${CMAKE_CURRENT_BINARY_DIR}/onnx)

    ADD_CUSTOM_COMMAND(OUTPUT  ${ONNX_PROTO_SRC}
                       COMMAND mkdir -p ${ONNX_PROTO_OUT_PATH}
                       COMMAND ${Protobuf_PROTOC_EXECUTABLE} --cpp_out=${ONNX_PROTO_OUT_PATH} --proto_path=${ONNX_PROTO_PATH} ${ONNX_PROTO_PATH}/onnx.proto
                       #COMMAND mv ${ONNX_PROTO_OUT_PATH}/onnx.pb.h ${ONNX_PROTO_PATH}/../include/
                       DEPENDS ${ONNX_PROTO_PATH}/onnx.proto)

    ADD_CUSTOM_TARGET(ONNX_SERIALIZER_TARGET DEPENDS ${ONNX_PROTO_SRC})

    include_directories(${ONNX_PROTO_OUT_PATH})

    list(APPEND ONNX_SERIALIZER_SRCS ${ONNX_PROTO_SRC})

    #message("ONNX serializer src: ${ONNX_SERIALIZER_SRCS}")
endif()



# MXNET SERIALIZER
if(BUILD_MXNET_SERIALIZER)
    FILE(GLOB_RECURSE MXNET_SERIALIZER_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/mxnet/*.cpp")
    #message("MXNet serializer src: ${MXNET_SERIALIZER_SRCS}")
endif()



# DARKNET SERIALIZER
if(BUILD_DARKNET_SERIALIZER)
    FILE(GLOB_RECURSE DARKNETT_SERIALIZER_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/darknet/*.cpp")
    #message("Darknet serializer src: ${DARKNETT_SERIALIZER_SRCS}")
endif()



# TENSORFLOW SERIALIZER
if(BUILD_TF_SERIALIZER)
    FILE(GLOB_RECURSE TF_SERIALIZER_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/tensorflow/*.cpp")

    list(APPEND TENGINE_LIB_SRCS ${serializer_src})

    # the generated pb.cc

    set(TF_PROTO_SRC           ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/graph.pb.cc
                               ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/function.pb.cc
                               ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/node_def.pb.cc
                               ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/op_def.pb.cc
                               ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/attr_value.pb.cc
                               ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/tensor.pb.cc
                               ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/tensor_shape.pb.cc
                               ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/types.pb.cc
                               ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/versions.pb.cc
                               ${CMAKE_CURRENT_BINARY_DIR}/tensorflow/resource_handle.pb.cc)

    set(TF_PROTO_PATH          ${CMAKE_CURRENT_SOURCE_DIR}/tensorflow)
    set(TF_PROTO_OUT_PATH      ${CMAKE_CURRENT_BINARY_DIR}/tensorflow)

    ADD_CUSTOM_COMMAND(OUTPUT  ${TF_PROTO_SRC}
                       COMMAND mkdir -p ${TF_PROTO_OUT_PATH}
                       COMMAND ${Protobuf_PROTOC_EXECUTABLE} --cpp_out=${TF_PROTO_OUT_PATH} --proto_path=${TF_PROTO_PATH} ${TF_PROTO_PATH}/graph.proto
                       COMMAND ${Protobuf_PROTOC_EXECUTABLE} --cpp_out=${TF_PROTO_OUT_PATH} --proto_path=${TF_PROTO_PATH} ${TF_PROTO_PATH}/function.proto
                       COMMAND ${Protobuf_PROTOC_EXECUTABLE} --cpp_out=${TF_PROTO_OUT_PATH} --proto_path=${TF_PROTO_PATH} ${TF_PROTO_PATH}/node_def.proto
                       COMMAND ${Protobuf_PROTOC_EXECUTABLE} --cpp_out=${TF_PROTO_OUT_PATH} --proto_path=${TF_PROTO_PATH} ${TF_PROTO_PATH}/op_def.proto
                       COMMAND ${Protobuf_PROTOC_EXECUTABLE} --cpp_out=${TF_PROTO_OUT_PATH} --proto_path=${TF_PROTO_PATH} ${TF_PROTO_PATH}/attr_value.proto
                       COMMAND ${Protobuf_PROTOC_EXECUTABLE} --cpp_out=${TF_PROTO_OUT_PATH} --proto_path=${TF_PROTO_PATH} ${TF_PROTO_PATH}/tensor.proto
                       COMMAND ${Protobuf_PROTOC_EXECUTABLE} --cpp_out=${TF_PROTO_OUT_PATH} --proto_path=${TF_PROTO_PATH} ${TF_PROTO_PATH}/tensor_shape.proto
                       COMMAND ${Protobuf_PROTOC_EXECUTABLE} --cpp_out=${TF_PROTO_OUT_PATH} --proto_path=${TF_PROTO_PATH} ${TF_PROTO_PATH}/types.proto
                       COMMAND ${Protobuf_PROTOC_EXECUTABLE} --cpp_out=${TF_PROTO_OUT_PATH} --proto_path=${TF_PROTO_PATH} ${TF_PROTO_PATH}/versions.proto
                       COMMAND ${Protobuf_PROTOC_EXECUTABLE} --cpp_out=${TF_PROTO_OUT_PATH} --proto_path=${TF_PROTO_PATH} ${TF_PROTO_PATH}/resource_handle.proto
                       #COMMAND mv ${TF_PROTO_OUT_PATH}/*.pb.h ${TF_PROTO_PATH}/../include/
    )

    ADD_CUSTOM_TARGET(TF_SERIALIZER_TARGET DEPENDS ${TF_PROTO_OUT_PATH})

    include_directories(${TF_PROTO_OUT_PATH})

    list(APPEND TF_SERIALIZER_SRCS ${TF_PROTO_SRC})
    #message("TF serializer src: ${TF_SERIALIZER_SRCS}")
endif()



# TFLITE SERIALIZER
if(BUILD_TFLITE_SERIALIZER)
    FILE(GLOB_RECURSE TFLITE_SERIALIZER_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/tf_lite/*.cpp")
    #message("TFLite serializer src: ${TFLITE_SERIALIZER_SRCS}")
endif()



# NCNN SERIALIZER
if(BUILD_NCNN_SERIALIZER)
    FILE(GLOB_RECURSE NCNN_SERIALIZER_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/ncnn/*.cpp")
    #message("NCNN serializer src: ${NCNN_SERIALIZER_SRCS}")
endif()



# MegEngine SERIALIZER
if(BUILD_MEGENGINE_SERIALIZER)
    if(NOT MEGENGINE_INSTALL_PATH)
        message(FATAL_ERROR "FATAL: If MegEngine model support was ENABLED, -DMEGENGINE_INSTALL_PATH must be set.")
    endif()

    include_directories(${MEGENGINE_INSTALL_PATH}/include)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/megengine)

    find_library(MEGENGINE_LIB NAMES megengine dnnl flatbuffers PATHS ${MEGENGINE_INSTALL_PATH}/lib)
    find_library(DNNL_LIB NAMES dnnl PATHS ${MEGENGINE_INSTALL_PATH}/lib)
    find_library(FLATBUFFERS_LIB NAMES flatbuffers PATHS ${MEGENGINE_INSTALL_PATH}/lib)

    FILE(GLOB_RECURSE MEGENGINE_SERIALIZER_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/megengine/*.cpp")
    #message("MegEngine serializer src: ${MEGENGINE_SERIALIZER_SRCS}")
endif()



if(BUILD_TENGINE_SERIALIZER)
    FILE(GLOB_RECURSE TENGINE_SERIALIZER_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/tengine/*.cpp")
    #message("Tengine serializer src: ${TENGINE_SERIALIZER_SRCS}")
endif()


# add convert tool files
FILE(GLOB MAIN_SERIALIZER_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")


# collection all serializer source files
list(APPEND FRAMEWORK_SERIALIZER_SRCS   ${CAFFE_SERIALIZER_SRCS})
list(APPEND FRAMEWORK_SERIALIZER_SRCS   ${ONNX_SERIALIZER_SRCS})
list(APPEND FRAMEWORK_SERIALIZER_SRCS   ${MXNET_SERIALIZER_SRCS})
list(APPEND FRAMEWORK_SERIALIZER_SRCS   ${DARKNETT_SERIALIZER_SRCS})
list(APPEND FRAMEWORK_SERIALIZER_SRCS   ${TF_SERIALIZER_SRCS})
list(APPEND FRAMEWORK_SERIALIZER_SRCS   ${TFLITE_SERIALIZER_SRCS})
list(APPEND FRAMEWORK_SERIALIZER_SRCS   ${NCNN_SERIALIZER_SRCS})
list(APPEND FRAMEWORK_SERIALIZER_SRCS   ${TENGINE_SERIALIZER_SRCS})
list(APPEND FRAMEWORK_SERIALIZER_SRCS   ${MAIN_SERIALIZER_SRCS})

if(BUILD_MEGENGINE_SERIALIZER)
    list(APPEND FRAMEWORK_SERIALIZER_SRCS   ${MEGENGINE_SERIALIZER_SRCS})
endif()


add_executable(convert_tool ${FRAMEWORK_SERIALIZER_SRCS})
target_link_libraries(convert_tool ${CMAKE_PROJECT_NAME} pthread dl m)

# OneFlow Serializer
if (BUILD_ONEFLOW_SERIALIZER)
    add_subdirectory(oneflow)
    target_link_libraries(convert_tool oneflow2tengine)
endif()

if(BUILD_CAFFE_SERIALIZER OR BUILD_ONNX_SERIALIZER OR BUILD_TF_SERIALIZER)
    if (Protobuf_LIBRARIES)
        target_link_libraries(convert_tool ${Protobuf_LIBRARIES})
    else()
        set(PROTOBUF_LIB_NAME Protobuf_LIBRARY_${CMAKE_BUILD_TYPE})
        target_link_libraries(convert_tool ${PROTOBUF_LIB_NAME})
    endif()
endif()

if(BUILD_MEGENGINE_SERIALIZER)
    target_link_libraries(convert_tool ${MEGENGINE_LIB} ${DNNL_LIB} ${FLATBUFFERS_LIB})
endif()

# enable code coverage test
if(BUILD_COVERAGE)
    SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fprofile-arcs -ftest-coverage")
    SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fprofile-arcs -ftest-coverage")    
    target_link_libraries(convert_tool ${CMAKE_PROJECT_NAME} -coverage -lgcov)
endif()

install (TARGETS convert_tool DESTINATION bin)
