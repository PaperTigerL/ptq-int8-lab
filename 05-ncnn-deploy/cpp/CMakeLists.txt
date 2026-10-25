cmake_minimum_required(VERSION 3.10)
project(cifar10_ncnn_eval CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------------------
# ncnn 位置
#
# ncnn 的头文件和库分散在两处，这是它从源码构建时的固有布局，别想当然合成一个：
#   NCNN_SRC_DIR    源码树里的 src/     —— net.h / mat.h / layer.h 等公开头
#   NCNN_BUILD_DIR  cmake 构建出的 build/src —— libncnn.a + layer_registry.h
#                                           等**生成**的头文件
# 换机器就 -DNCNN_SRC_DIR=... -DNCNN_BUILD_DIR=... 覆盖，别改这个文件。
# ---------------------------------------------------------------------------
if(NOT DEFINED NCNN_SRC_DIR)
    set(NCNN_SRC_DIR "$ENV{HOME}/ncnn-build/ncnn/src")
endif()
if(NOT DEFINED NCNN_BUILD_DIR)
    set(NCNN_BUILD_DIR "$ENV{HOME}/ncnn-build/ncnn/build")
endif()

find_library(NCNN_LIBRARY ncnn PATHS "${NCNN_BUILD_DIR}/src" NO_DEFAULT_PATH)
find_path(NCNN_INCLUDE_DIR net.h PATHS "${NCNN_SRC_DIR}" NO_DEFAULT_PATH)

if(NOT NCNN_LIBRARY OR NOT NCNN_INCLUDE_DIR)
    message(FATAL_ERROR
        "找不到 ncnn。\n"
        "  NCNN_SRC_DIR=${NCNN_SRC_DIR} -> ${NCNN_INCLUDE_DIR}\n"
        "  NCNN_BUILD_DIR=${NCNN_BUILD_DIR} -> ${NCNN_LIBRARY}\n"
        "先在 WSL 里编译 ncnn，或用 -DNCNN_SRC_DIR=/-DNCNN_BUILD_DIR= 指过去。")
endif()

message(STATUS "ncnn 头文件 : ${NCNN_INCLUDE_DIR}")
message(STATUS "ncnn 生成头 : ${NCNN_BUILD_DIR}/src")
message(STATUS "ncnn 库     : ${NCNN_LIBRARY}")

find_package(Threads REQUIRED)
# ncnn 内部用 OpenMP 做多线程并行（GOMP_parallel 就是它的入口）。
# 只链 pthread 不够 —— 那正是 "undefined reference to GOMP_parallel" 的原因。
find_package(OpenMP REQUIRED)

add_executable(cifar10_eval
    main.cpp
    cifar10.cpp
    evaluator.cpp
)
target_include_directories(cifar10_eval PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${NCNN_INCLUDE_DIR}          # net.h / mat.h
    ${NCNN_BUILD_DIR}/src        # layer_registry.h 等生成文件
)
target_link_libraries(cifar10_eval PRIVATE
    ${NCNN_LIBRARY}
    Threads::Threads
    OpenMP::OpenMP_CXX
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(cifar10_eval PRIVATE -O2)
endif()
