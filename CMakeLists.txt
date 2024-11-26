cmake_minimum_required(VERSION 3.12)
project(zdlf)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

set(CMAKE_BUILD_TYPE Release)

add_subdirectory(pybind11)

#find_package(OpenMP)

pybind11_add_module(tensor MODULE tensorpybind.h)

#if (OpenMP_CXX_FOUND)
#    include_directories(OpenMP_CXX_INCLUDE_DIRS)
#    target_link_libraries(zdlf PUBLIC OpenMP::OpenMP_CXX)
#endif()

