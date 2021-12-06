cmake_minimum_required(VERSION 3.14)
project(BKTree)

add_library(BKTreeCpp STATIC library.cc library.h)

set(CMAKE_BUILD_TYPE "Release")
Include(FetchContent)

FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.5.0)
FetchContent_MakeAvailable(pybind11)
pybind11_add_module(BKTree MODULE BKTree.cc)
target_sources(BKTree PRIVATE library.cc library.h)
target_link_libraries(BKTree PRIVATE SymSpellCpp)
target_compile_options(BKTree PRIVATE
        $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native -mtune=native -fvisibility=hidden -flto>)
message(STATUS "Build for Python = " ${BUILD_FOR_PYTHON})