#include "gpu_error_check.cuh"
#include "planning_state.cuh"
#include "generate_trajectories.cuh"
#include "vector.cuh"
#include "trajectory_msgs/JointTrajectory.h"
#include "trajectory_msgs/JointTrajectoryPoint.h"

#include <cuda.h>

PlanningState::PlanningState(unsigned int num_initial_trajectories_, unsigned int num_noise_vectors_, unsigned int num_waypoints_, unsigned int waypoint_dim_):
    num_initial_trajectories(num_initial_trajectories_), num_noise_vectors(num_noise_vectors_), num_waypoints(num_waypoints_), waypoint_dim(waypoint_dim_), _have_goal(false), trajectories(num_initial_trajectories_* num_waypoints_ * waypoint_dim_), noise_vectors(num_initial_trajectories_ * num_noise_vectors_ * waypoint_dim_), noisy_trajectories(num_initial_trajectories_ * num_noise_vectors_ * num_waypoints_ * waypoint_dim_), rngs(num_initial_trajectories_ * waypoint_dim_ * std::max(num_noise_vectors_, num_waypoints_)), velocities(num_initial_trajectories_ * num_noise_vectors_ * num_waypoints_ * waypoint_dim_), accelerations(num_initial_trajectories_ * num_noise_vectors_ * num_waypoints_ * waypoint_dim_), smoothness(num_initial_trajectories_ * num_noise_vectors_), scores(num_initial_trajectories_ * (num_noise_vectors_ + 1)), update_vectors(num_initial_trajectories_ * waypoint_dim_), flag(1), found_better(num_initial_trajectories_) {
    auto num_rngs = num_initial_trajectories * waypoint_dim * std::max(num_noise_vectors, num_waypoints);
    init_cudarand<<<ceil(double(num_rngs)/double(512)), 512>>>(rngs.data(), num_rngs);
    if(cudaPeekAtLastError() != cudaSuccess) {
        std::cerr << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
        ROS_ASSERT(false);
    }
    if(cudaDeviceSynchronize() != cudaSuccess) {
        ROS_ASSERT(false);
    }
}

std::vector<float> PlanningState::plan_lerp() const {
    std::vector<float> result(num_waypoints * waypoint_dim);
    std::vector<float> step_sizes(waypoint_dim);
    for(size_t dim = 0; dim < waypoint_dim; ++dim) {
        step_sizes[dim] = (goal_state[dim] - current_state[dim]) / num_waypoints;
    }
    for(size_t waypoint = 0; waypoint < num_waypoints; ++waypoint) {
        for(size_t dim = 0; dim < waypoint_dim; ++dim) {
            result[waypoint*waypoint_dim + dim] = current_state[dim] + waypoint*step_sizes[dim];
        }
    }
    return result;
}

void PlanningState::plan() {
    if(!_have_goal) return;
    synchronize();
    std::vector<float> lerp = plan_lerp();
    dim3 gridDim(num_initial_trajectories);
    dim3 blockDim(num_waypoints * num_noise_vectors * waypoint_dim);
    if(cudaMemcpyToSymbol(initial_waypoint, current_state.data(), current_state.size() * sizeof(float), 0, cudaMemcpyHostToDevice) != cudaSuccess) {
        ROS_ASSERT(false);
    }
    if(cudaMemcpyToSymbol(final_waypoint, goal_state.data(), goal_state.size() * sizeof(float), 0, cudaMemcpyHostToDevice) != cudaSuccess) {
        ROS_ASSERT(false);
    }
    if(cudaMemcpyToSymbol(initial_trajectory, lerp.data(), lerp.size() * sizeof(float), 0, cudaMemcpyHostToDevice) != cudaSuccess) {
        ROS_ASSERT(false);
    }
    flag[0] = false;
    optimize_trajectories<<<gridDim, blockDim>>>(trajectories.data(), noise_vectors.data(), noisy_trajectories.data(), rngs.data(), velocities.data(), accelerations.data(), smoothness.data(), scores.data(), update_vectors.data(), found_better.data(), flag.data(), waypoint_dim*std::max(num_waypoints, num_noise_vectors), num_waypoints, waypoint_dim, num_noise_vectors, deltaT);
    if(cudaPeekAtLastError() != cudaSuccess) { //make sure the kernel launched
        std::cerr << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
        ROS_ASSERT(false);
    }
    planning_trajectory = true;
}

void PlanningState::synchronize() {
    //sync with GPU
    flag[0] = true;
    if(cudaPeekAtLastError() != cudaSuccess) {
        ROS_ASSERT(false);
    }
    if(cudaDeviceSynchronize() != cudaSuccess) {
        ROS_ASSERT(false);
    }
}

void PlanningState::publish_trajectory(ros::Publisher& pub) {
    if(!planning_trajectory) return;
    synchronize();

    // find lowest cost trajectory
    size_t best_trajectory_index;
    float lowest_cost = std::numeric_limits<float>::infinity();
    for(size_t i = 0; i < num_initial_trajectories * (num_noise_vectors + 1); ++i) {
        if(scores[i] < lowest_cost) {
            lowest_cost = scores[i];
            best_trajectory_index = i;
        }
    }
    std::cout << "best trajectory cost: " << lowest_cost << std::endl;
    std::cout << "best trajectory index: " << best_trajectory_index << std::endl;

    bool publish_initial_trajectory = best_trajectory_index % (num_noise_vectors + 1) == 0;
    std::vector<float> lerp;
    std::vector<float> lerp_vels;
    if(publish_initial_trajectory) {
        std::cout << "publishing initial trajectory" << std::endl;
        lerp = plan_lerp();
        lerp_vels.resize(num_waypoints*waypoint_dim);
        for(size_t i = 0; i < num_waypoints-1; ++i) {
            for(size_t d = 0; d < waypoint_dim; ++d) {
                lerp_vels[i*waypoint_dim + d] = (lerp[(i+1)*waypoint_dim + d] - lerp[i*waypoint_dim + d]) / deltaT;
            }
        }
        for(size_t d = 0; d < waypoint_dim; ++d) {
            lerp_vels[(num_waypoints-1)*waypoint_dim + d] = 0;
        }
    }
    float* best_trajectory = publish_initial_trajectory ? lerp.data() : noisy_trajectories.data() + (best_trajectory_index * num_waypoints * waypoint_dim);
    float* best_velocity = publish_initial_trajectory ? lerp_vels.data() : velocities.data() + (best_trajectory_index * num_waypoints * waypoint_dim);

    trajectory_msgs::JointTrajectory msg;
    msg.header.stamp = ros::Time::now();
    //set all but the last point and velocity
    trajectory_msgs::JointTrajectoryPoint point;
    //attach initial state
    for(size_t i = 0; i < waypoint_dim; ++i) {
        point.positions.push_back(current_state[i]);
        point.velocities.push_back((best_trajectory[i] - current_state[i]) / deltaT);
    }
    point.time_from_start = ros::Duration(0);
    msg.points.push_back(point);
    point.positions.clear();
    point.velocities.clear();
    for(size_t i = 0; i < num_waypoints; ++i) {
        for(size_t j = 0; j < waypoint_dim; ++j) {
            point.positions.push_back(best_trajectory[i*waypoint_dim + j]);
            point.velocities.push_back(best_velocity[(i+1)*waypoint_dim + j]); //due to how velocities are computed on GPU (look-back vs look-forward)
        }
        point.time_from_start = ros::Duration(deltaT * i);
        msg.points.push_back(std::move(point));
    }

    for(size_t j = 0; j < waypoint_dim; ++j) {
        point.positions.push_back(goal_state[j]);
        point.velocities.push_back(0); //stop at last point
    }
    msg.points.push_back(std::move(point));
    pub.publish(std::move(msg));
    _have_goal = false;
}
