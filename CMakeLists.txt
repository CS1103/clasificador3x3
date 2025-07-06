cmake_minimum_required(VERSION 3.10)
project(Clasificador3x3)

set(CMAKE_CXX_STANDARD 20)

include_directories(include)
include_directories(data)
include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB SOURCES "src/*.cpp")

add_executable(clasificador3x3 ${SOURCES})
