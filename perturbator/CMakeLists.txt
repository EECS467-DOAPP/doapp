cmake_minimum_required(VERSION 3.11)

project(perturbator VERSION 0.1.0 LANGUAGES CUDA)
set(CMAKE_CUDA_STANDARD 11)
#set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} -dc)
add_library(perturbator SHARED random.cu trajectory.cu vector.cu)

add_library(trajectory_gen STATIC generate_trajectories.cu)
set_target_properties(trajectory_gen PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_rand test_rand.cu)
target_link_libraries(test_rand PRIVATE perturbator)

add_executable(test test.cu)
target_link_libraries(test PRIVATE perturbator)

add_executable(main main.cu)
target_link_libraries(main PRIVATE ${CUDA_curand_LIBRARY}) # set by the cuda package

add_executable(test_trajectory_gen generate_trajectories.test.cu)
set_target_properties(test_trajectory_gen PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_trajectory_gen PRIVATE trajectory_gen ${CUDA_curand_LIBRARY})
