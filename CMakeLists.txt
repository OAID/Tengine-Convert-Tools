cmake_minimum_required(VERSION 3.6)

# Disable inplace builds to prevent source tree corruption.
IF (" ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}")
    MESSAGE (FATAL_ERROR "FATAL: Building inplace are not allowed. You should create a separate directory for Building.")
ENDIF ()

# set cmake_install_prefix path
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
endif()
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

# default budilding type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Choose the type of build" FORCE)
endif()

# project name
project(tengine-serializer C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CXX_STANDARD_REQUIRED ON)

# building flags
#target_compile_definitions(operator,"--allow-shlib-undefined")
add_definitions(-Wl,-Bsymbolic)
add_definitions(-Wl,-Bsymbolic-functions)
add_definitions(-Wno-unused-command-line-argument)
add_definitions(-funroll-loops)
add_definitions(-Wno-deprecated-register)
add_definitions(-Wno-overloaded-virtual)
add_definitions(-fPIC)
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -matomics")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -pthread -matomics")
endif()

# paser options
option(BUILD_CAFFE_SERIALIZER       "Caffe serializer"          ON)
option(BUILD_ONNX_SERIALIZER        "ONNX serializer"           ON)
option(BUILD_ONEFLOW_SERIALIZER     "OneFlow serializer"        ON)
option(BUILD_MXNET_SERIALIZER       "MXNet serializer"          ON)
option(BUILD_DARKNET_SERIALIZER     "Darknet serializer"        ON)
option(BUILD_TF_SERIALIZER          "TensorFlow serializer"     ON)
option(BUILD_TFLITE_SERIALIZER      "TFLite serializer"         ON)
option(BUILD_NCNN_SERIALIZER        "NCNN serializer"           ON)
option(BUILD_MEGENGINE_SERIALIZER   "MegEngine serializer"      OFF)
option(BUILD_TENGINE_SERIALIZER     "Tengine serializer"        ON)

# some basic options
option(BUILD_COVERAGE "build for coverage" OFF)

if (BUILD_ONEFLOW_SERIALIZER AND (${CMAKE_VERSION} VERSION_LESS "3.7.0"))
    message(FATAL_ERROR "Please upgrade your cmake (maybe by \"pip3 install -U cmake\") or disable OneFlow serializer by \"cmake -DBUILD_ONEFLOW_SERIALIZER=OFF ..\"")
endif()

# add all header file folder
include_directories(include)
include_directories(core/include)
include_directories(driver/include)
include_directories(driver/cpu)
include_directories(executor/include)
include_directories(executor/operator/include)
include_directories(operator/include)
include_directories(serializer/include)
include_directories(serializer/include/tengine/v2)

# add
file(GLOB_RECURSE CORE_SRC       "${CMAKE_CURRENT_SOURCE_DIR}/core/lib/*.cpp"          "${CMAKE_CURRENT_SOURCE_DIR}/core/lib/logger/*.cpp")
file(GLOB_RECURSE CPU_DRIVER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/driver/cpu/*.cpp"        "${CMAKE_CURRENT_SOURCE_DIR}/driver/plugin/init.cpp")
file(GLOB_RECURSE COMMON_SRC     "${CMAKE_CURRENT_SOURCE_DIR}/executor/engine/*.cpp"   "${CMAKE_CURRENT_SOURCE_DIR}/executor/lib/*.cpp"       "${CMAKE_CURRENT_SOURCE_DIR}/executor/plugin/*.cpp")
file(GLOB_RECURSE OPERATOR_SRC   "${CMAKE_CURRENT_SOURCE_DIR}/operator/plugin/*.cpp"   "${CMAKE_CURRENT_SOURCE_DIR}/operator/operator/*.cpp")
file(GLOB_RECURSE SERIALIZER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/serializer/plugin/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/serializer/tengine/*.c"   "${CMAKE_CURRENT_SOURCE_DIR}/serializer/tengine/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/serializer/tengine/v1/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/serializer/tengine/v2/*.cpp")

# add exec
add_library(${CMAKE_PROJECT_NAME} STATIC ${CORE_SRC} ${CPU_DRIVER_SRC} ${COMMON_SRC} ${OPERATOR_SRC} ${SERIALIZER_SRC})

# add sub folder
add_subdirectory(tools)
