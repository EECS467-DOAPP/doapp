cmake_minimum_required(VERSION 3.11)

project(perturbator VERSION 0.1.0 LANGUAGES CUDA)
set(CMAKE_CUDA_STANDARD 11)
find_package(catkin REQUIRED COMPONENTS roscpp std_msgs sensor_msgs)
#catkin_package(
#    INCLUDE_DIRS include
#    CATKIN_DEPENDS roscpp std_msgs sensor_msgs
#    )

include_directories(include ${catkin_INCLUDE_DIRS})

add_library(perturbator SHARED src/matrix.cu src/random.cu src/trajectory.cu src/vector.cu)

add_library(trajectory_gen STATIC src/generate_trajectories.cu) # has to be static for separable compilation to link correctly (idk why)
set_target_properties(trajectory_gen PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(trajectory_gen PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:--generate-line-info -G>)

add_library(common STATIC src/common.cu) # has to be static for separable compilation to link correctly (idk why)
set_target_properties(common PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(common PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:--generate-line-info -G>)

add_executable(test_rand src/test_rand.cu)
target_link_libraries(test_rand PRIVATE perturbator)

add_executable(test_matrix src/test_matrix.cu)
target_link_libraries(test_matrix PRIVATE perturbator)

add_executable(main src/main.cu)
target_link_libraries(main PRIVATE ${CUDA_curand_LIBRARY} trajectory_gen) #${catkin_LIBRARIES})

add_executable(test_trajectory_gen src/generate_trajectories.test.cu)
set_target_properties(test_trajectory_gen PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_trajectory_gen PRIVATE trajectory_gen common ${CUDA_curand_LIBRARY})
target_compile_options(test_trajectory_gen PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:--generate-line-info -G>)

add_executable(device_query src/device_query.cu)

