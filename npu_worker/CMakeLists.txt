cmake_minimum_required(VERSION 3.16)
project(a7a_yolo_worker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(a7a_yolo_worker a7a_yolo_worker.cpp)
