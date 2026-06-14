cmake_minimum_required(VERSION 3.10)

project(cpp_astar_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(cpp_astar_demo
    main.cpp
    astar.cpp
)
