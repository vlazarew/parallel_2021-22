cmake_minimum_required(VERSION 3.21)
project(task4 CUDA)

set(CMAKE_CUDA_STANDARD 11)

add_executable(task4 main.cu)

set_target_properties(task4 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
