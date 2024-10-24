cmake_minimum_required(VERSION 3.10)

# Set the project name
project(TreehornCV VERSION 1.0)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Add the executable
add_executable(test src/cpp/test.cpp src/cpp/opti_vibe.cpp)

# Include directories
target_include_directories(test PRIVATE include)

# Link OpenCV libraries
target_link_libraries(test PRIVATE ${OpenCV_LIBS})

# Set C++ standard to 20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
