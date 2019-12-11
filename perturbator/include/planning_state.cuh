#ifndef PLANNING_STATE_H
#define PLANNING_STATE_H
#include <vector>
#include "common.cuh"
#include "vector.cuh"
#include "ros/ros.h"
#include "sensor_msgs/JointState.h"
#include <curand_kernel.h>
#include <iostream> //TODO: remove this
class PlanningState {
public:
    PlanningState(unsigned int num_initial_trajectories_, unsigned int num_noise_vectors_, unsigned int num_waypoints_, unsigned int waypoint_dim_);

    //update arm state
    void updateArmState(const sensor_msgs::JointState::ConstPtr& msg) {
        std::cout << "updated arm state" << std::endl;
        current_state = std::vector<float>(msg->position.begin(), msg->position.end());
        ROS_ASSERT(current_state.size() == doapp::num_joints);
    }

    //update new goal
    void updateGoalState(const sensor_msgs::JointState::ConstPtr& msg) {
        _have_goal = true;
        goal_state = std::vector<float>(msg->position.begin(), msg->position.end());
        ROS_ASSERT(goal_state.size() == doapp::num_joints);
    }

    bool have_goal() {
        return _have_goal;
    }

    void plan();

    void publish_trajectory(ros::Publisher& pub);
    std::vector<float> plan_lerp() const;

private:
    void synchronize();
    const float deltaT = 1; //one second, for now
    const float viable_score_max = 100; //literally have no clue what this should be, other than the cost of one obstacle collision
    bool planning_trajectory = false, _have_goal;
    unsigned int num_initial_trajectories, num_noise_vectors, num_waypoints, waypoint_dim;
    std::vector<float> goal_state{doapp::num_joints};
    std::vector<float> current_state{doapp::num_joints};
    doapp::Vector<float, doapp::Dynamic> trajectories;
    doapp::Vector<float, doapp::Dynamic> noise_vectors;
    doapp::Vector<float, doapp::Dynamic> noisy_trajectories;
    doapp::Vector<curandState, doapp::Dynamic> rngs;
    doapp::Vector<float, doapp::Dynamic> velocities;
    doapp::Vector<float, doapp::Dynamic> accelerations;
    doapp::Vector<float, doapp::Dynamic> smoothness;
    doapp::Vector<float, doapp::Dynamic> scores;
    doapp::Vector<float, doapp::Dynamic> update_vectors;
    doapp::Vector<bool, doapp::Dynamic> flag;
    doapp::Vector<bool, doapp::Dynamic> found_better;
};
#endif
