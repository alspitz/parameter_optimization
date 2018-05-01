cmake_minimum_required(VERSION 2.8.3)
project(quad_model)

find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS})
add_library(quad_model model.cpp)

add_subdirectory(../../sources/pybind11 pybind11)

pybind11_add_module(quad_model_py model.cpp model_python.cpp)
