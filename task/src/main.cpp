#include <ros/ros.h>
#include "trajectory_msgs/JointTrajectory.h"
#include "sensor_msgs/JointState.h"
#include "std_msgs/Empty.h"
#include <vector>
#include <iostream>

constexpr double DEG2RAD = (2*M_PI/360.0);

void update_state(const sensor_msgs::JointState::ConstPtr& msg);
void publish_new_goal();
void go(const std_msgs::Empty::ConstPtr& msg);

ros::Publisher pub;
int main(int argc, char** argv) {
    ros::init(argc, argv, "task");

    ros::NodeHandle n;

    ros::Duration sleeper(1);

    ros::Subscriber joint_state_sub = n.subscribe("joint_state", 1, &update_state);
    ros::Subscriber go_sub = n.subscribe("/go", 1, &go);
    pub = n.advertise<sensor_msgs::JointState>("goal_state", 1);
    publish_new_goal();
    while(ros::ok()) {
        sleeper.sleep();
        ros::spinOnce();
    }
}

std::vector<double> current_state;
std::vector<std::vector<double>> goals = {
    {0.0, 0.0, 0.0, 0.0},
    {0.0, 1.0, 0.0, 0.5},
    //{117.14 * DEG2RAD, 25.78 * DEG2RAD, 52.26 * DEG2RAD, 70.79 * DEG2RAD},
    {1.5, 1.5, 1.5, 1.5},
    {0.0, 0.0, 1.0, -0.5},
    {0.0, -1.0, 1.5, -1.5},
    {0.0, -0.0, 0.5, -0.5},
    {0.0, 0.0, 0.0, 0.0}
};
std::size_t goal_index = 0;
bool new_goal = true;

void update_state(const sensor_msgs::JointState::ConstPtr& msg) {
    current_state = msg->position;
    std::vector<double>& goal = goals[goal_index];
    double error = 0;
    static double max_error = 0.1; //no clue what this should be
    for(size_t i = 0; i < goal.size() && error < max_error; ++i) {
        error += std::abs(current_state[i] - goal[i]);
    }
    std::cout << "Error: " << error << std::endl;
    if(error > max_error) return;
    //we are at the goal position, time to flip it
    goal_index = ++goal_index % goals.size();
    new_goal = true;
    publish_new_goal();
}

void publish_new_goal() {
    std::vector<double>& goal = goals[goal_index];
    sensor_msgs::JointState msg;
    msg.header.stamp = ros::Time::now();
    msg.position = goal;
    pub.publish(msg);
    std::cout << "published" << std::endl;
    new_goal = false;
}

void go(const std_msgs::Empty::ConstPtr& msg) {
    new_goal = true;
    goal_index = ++goal_index % goals.size();
    publish_new_goal();
}
