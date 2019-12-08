#include "common.cuh"

#define MIN_JOINT_ANGLES {1024.0f, 1024.0f, 1024.0f, 0.0f, 0.0f}
#define MAX_JOINT_ANGLES {3072.0f, 3072.0f, 3072.0f, 2013.0f, 2013.0f}

namespace doapp {

__constant__ float min_joint_angles[num_joints] = MIN_JOINT_ANGLES;
__constant__ float max_joint_angles[num_joints] = MAX_JOINT_ANGLES;
float host_min_joint_angles[num_joints] = MIN_JOINT_ANGLES;
float host_max_joint_angles[num_joints] = MAX_JOINT_ANGLES;

} // namespace doapp
