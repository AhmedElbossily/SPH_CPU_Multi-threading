cmake_minimum_required(VERSION 3.12)
project(SPH_CPU_Multithreading)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the include directories
include_directories(include)

# Add flag for using multithreading
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++20 -pthread")

# Add your source files here
file(GLOB SOURCES "src/*.cpp")

# Create the executable
add_executable(${PROJECT_NAME} ${SOURCES})