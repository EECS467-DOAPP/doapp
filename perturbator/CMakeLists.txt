cmake_minimum_required(VERSION 3.11)

project(perturbator VERSION 0.1.0 LANGUAGES CUDA)

add_library(perturbator SHARED random.cu trajectory.cu vector.cu)

add_executable(test test.cu)
target_link_libraries(test PRIVATE perturbator)
