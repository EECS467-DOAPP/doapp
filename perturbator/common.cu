#include "common.cuh"
//I know I know, it's duplicated code. If you wanna have it be non duplicated, feel free to use a macro. Oh, what's that? You don't wanna use a macro? That's what I thought.
__constant__ float doapp::min_joint_angles[num_joints] = {1024.0f, 1024.0f, 1024.0f, 0.0f, 0.0f};
__constant__ float doapp::max_joint_angles[num_joints] = {3072.0f, 3072.0f, 3072.0f, 2013.0f, 2013.0f};
float doapp::host_min_joint_angles[num_joints] = {1024.0f, 1024.0f, 1024.0f, 0.0f, 0.0f};
float doapp::host_max_joint_angles[num_joints] = {3072.0f, 3072.0f, 3072.0f, 2013.0f, 2013.0f};
