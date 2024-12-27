# Minimum required version of CMake
cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80")

# Project name and supported languages
project(MyCudaProject LANGUAGES CXX CUDA)

# Set the C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 11)

# Check if CUDA is available
find_package(CUDA REQUIRED)

# Add CUDA include paths
include_directories(${CUDA_INCLUDE_DIRS})

# Add CUDA library paths
link_directories(${CUDA_LIBRARY_DIRS})

# Add all source files for the project
file(GLOB SOURCES "*.cpp" "*.cu")


# Define the target executable name
add_executable(my_gemm ${SOURCES})

# Link CUDA and cuBLAS libraries
target_link_libraries(my_gemm PRIVATE cuda cublas)

# Enable CUDA build options
set_target_properties(my_gemm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "60;70;75;80" # Set supported GPU architectures
)

# Add debug information (optional)
target_compile_options(my_gemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
