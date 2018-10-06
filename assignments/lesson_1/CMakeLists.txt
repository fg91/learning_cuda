# "Copyright 2018 <Fabio M. Graetz>"
cmake_minimum_required(VERSION 2.8)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

file(GLOB h *.hpp *.h)
file(GLOB cu *.cu)
file(GLOB cpp *.cpp)
set(hw1_files main.cpp)

CUDA_ADD_EXECUTABLE(hw1 ${hw1_files} ${h} ${cu} ${cpp})
target_link_libraries(hw1 ${OpenCV_LIBS})
