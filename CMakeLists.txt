cmake_minimum_required(VERSION 3.14)
project(Project)

set(CMAKE_CXX_STANDARD 17)

# --- Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Project
        main.cpp
        src/red-eyes.cpp
        src/red-eyes.h
)

target_link_libraries(Project ${OpenCV_LIBS})
