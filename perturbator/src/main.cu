#include "common.cuh"
#include "cpu_distance_grid.cuh"
#include "generate_trajectories.cuh"
#include "gpu_error_check.cuh"
#include "planning_state.cuh"
#include "vector.cuh"

#include "ros/ros.h"
#include "trajectory_msgs/JointTrajectory.h"

#include <cfloat>


int main(int argc, char** argv) {
    ros::init(argc, argv, "motion_planner");
    ros::NodeHandle node_handle;

    const doapp::cpu_distance_grid::Dimensions grid_dims = {100, 100, 100, 0.01f};

    unsigned int k = 1000, n = 50, m = 50, d = doapp::num_joints; //TODO: have m be calculated from 1024/(n*d)
    m = 1024 / n / d;
    //also TODO: have k, n, m, d be grabbed from ParamServer
    std::cout << "Running with k = " << k << ", n = " << n << ", m = " << m << std::endl;
    PlanningState planning_state(k, m, n, d, grid_dims);

    ros::Subscriber state_subscriber = node_handle.subscribe("joint_state", 1, &PlanningState::updateArmState, &planning_state);
    ros::Subscriber goal_subscriber = node_handle.subscribe("goal_state", 1, &PlanningState::updateGoalState, &planning_state); //don't really want to buffer old requests
    //no clue what loop_rate should be
    ros::Rate loop_rate(1);
    ros::Publisher trajectory_publisher = node_handle.advertise<trajectory_msgs::JointTrajectory>("/joint_trajectory", 2);
    while(ros::ok()) {
        ros::spinOnce();
        if(planning_state.have_goal()) {
            std::cout << "starting to plan" << std::endl;
            planning_state.plan();
        }
        loop_rate.sleep();
        //TODO: see if we have to update the current state of the arm here so then publishing doesn't cause the arm to "go back in time"
        //TODO: we should be planning in advance, so this shouldn't happen
        if(planning_state.have_goal()) {
            std::cout << "going to publish" << std::endl;
            planning_state.publish_trajectory(trajectory_publisher);
        }
    }
}
