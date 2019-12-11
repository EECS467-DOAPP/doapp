cmake_minimum_required(VERSION 3.11)

project(perturbator VERSION 0.1.0 LANGUAGES CXX CUDA)
set(CMAKE_CUDA_STANDARD 11)

find_package(Threads REQUIRED)

add_library(perturbator SHARED cpu_distance_grid.cu)
target_link_libraries(perturbator PRIVATE Threads::Threads)
target_include_directories(perturbator PUBLIC nanoflann/include)
set_target_properties(perturbator PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_SEPARABLE_COMPILATION ON)

add_library(trajectory_gen STATIC generate_trajectories.cu) # has to be static for separable compilation to link correctly (idk why)
set_target_properties(trajectory_gen PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(trajectory_gen PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:--generate-line-info -G>)

add_library(common STATIC common.cu) # has to be static for separable compilation to link correctly (idk why)
set_target_properties(common PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(common PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:--generate-line-info -G>)

add_executable(test_rand test_rand.cu)
target_link_libraries(test_rand PRIVATE perturbator)

add_executable(test_matrix test_matrix.cu)
target_link_libraries(test_matrix PRIVATE perturbator)

add_executable(test_distance_grid test_distance_grid.cu)
target_link_libraries(test_distance_grid PRIVATE perturbator)
set_target_properties(test_distance_grid PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_SEPARABLE_COMPILATION ON)

add_executable(main main.cu)
target_link_libraries(main PRIVATE ${CUDA_curand_LIBRARY}) # set by the cuda package

add_executable(test_trajectory_gen generate_trajectories.test.cu)
set_target_properties(test_trajectory_gen PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_trajectory_gen PRIVATE trajectory_gen common ${CUDA_curand_LIBRARY})
target_compile_options(test_trajectory_gen PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:--generate-line-info -G>)

add_executable(device_query device_query.cu)
