cmake_minimum_required(VERSION 3.29)
project(ml C)

set(CMAKE_C_STANDARD 11)

include_directories(.)

add_executable(ml
        xor.c
    nn.h)

target_link_libraries(ml m)
