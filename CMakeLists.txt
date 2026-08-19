# Minimum CMake version
cmake_minimum_required(VERSION 3.18)

# Project name, plus the languages used.
project(counter_uas LANGUAGES CXX CUDA)

# C++17, as specified in the project design
set(CMAKE_CXX_STANDARD 17)
# REQUIRED means fail loudly rather than silently
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Orin's GPU is Ampere, compute capability 8.7 -> "87".
# CMake equivalent of nvcc -arch=sm_87.
# Wrong value gives "no kernel image available for execution",
# which means the binary holds code for a GPU you don't have.
set(CMAKE_CUDA_ARCHITECTURES 87)

# Writes build/compile_commands.json describing how each file is compiled.
# VS Code's C++ extension reads it to resolve includes, so headers like
# <cuda_runtime.h> stop showing false red underlines.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# More executables and libraries get added here as the project grows.
add_executable(smoke src/app/smoke.cu)