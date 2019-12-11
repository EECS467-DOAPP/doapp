#ifndef PLANNER_CUH
#define PLANNER_CUH

#include "vector.cuh"

#include <cstddef>
#include <chrono>

namespace doapp {
namespace detail {

void do_plan(std::size_t k, std::size_t m, std::size_t n, std::size_t d,
             float *output, const float *initial_position,
             const float *initial_velocity,
             const JointPositionBound *position_bounds,
             const JointVelocityBound *velocity_bounds,
             const JointAccelerationBound *acceleration_bounds,
             std::chrono::steady_clock::duration timeout);

} // namespace detail

struct JointPositionBound {
    float min;
    float max;
};

struct JointVelocityBound {
    float min;
    float max;
};

struct JointAccelerationBound {
    float min;
    float max;
};

template <std::size_t NumJoints>
struct PlannerParams {
    std::size_t num_trajectories;
    std::size_t num_noise_vectors;
    std::size_t num_waypoints;

    Vector<float, NumJoints> initial_position;
    Vector<float, NumJoints> initial_velocity;

    Vector<JointPositionBound, NumJoints> position_bounds;
    Vector<JointVelocityBound, NumJoints> velocity_bounds;
    Vector<JointAccelerationBound, NumJoints> acceleration_bounds;

    std::chrono::steady_clock::duration timeout;
};

template <std::size_t NumJoints>
Vector<Vector<float, NumJoints>, Dynamic> plan(const PlannerParams &params) {
    Vector<float, Dynamic> output(params.num_waypoints * NumJoints);

    do_plan(params.num_trajectories, params.num_noise_vectors,
            params.num_waypoints, NumJoints, output.data(),
            initial_position.data(), initial_velocity.data(),
            position_bounds.data(), velocity_bounds.data(),
            acceleration_bounds.data(), timeout);

    Vector<Vector<float, NumJoints>> real_output(params.num_waypoints);

    for (std::size_t i = 0; i < params.num_waypoints; ++i) {
        for (std::size_t j = 0; j < NumJoints; ++j) {
            real_output[i][j] = output[i * NumJoints + j];
        }
    }

    return real_output;
}

} // namespace doapp

#endif
