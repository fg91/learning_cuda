# "Copyright 2018 <Fabio M. Graetz>"
cmake_minimum_required(VERSION 2.8)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

file(GLOB h *.hpp *.h)
file(GLOB cu *.cu)
file(GLOB cpp *.cpp)
set(hw5_files main.cu)

CUDA_ADD_EXECUTABLE(hw5 ${hw5_files} ${h} ${cu} ${cpp})
target_link_libraries(hw5 ${OpenCV_LIBS})
