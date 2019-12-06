cmake_minimum_required(VERSION 3.11)

project(perturbator VERSION 0.1.0 LANGUAGES CUDA)

add_library(perturbator SHARED matrix.cu random.cu trajectory.cu vector.cu)

add_executable(test_rand test_rand.cu)
target_link_libraries(test_rand PRIVATE perturbator)

add_executable(test_matrix test_matrix.cu)
target_link_libraries(test_matrix PRIVATE perturbator)

add_executable(main main.cu)
target_link_libraries(main PRIVATE ${CUDA_curand_LIBRARY}) # set by the cuda package
