#ifndef __DRIVER_H
#define __DRIVER_H

#define NUM_MOTORS 6       // number of motors on Rexarm
#define NUM_TYPE_SWITCH 3  // motor ID that switches from MX-28 to AX-12A
#include <dynamixel_driver/MotorCommand.h>
#include <dynamixel_workbench_msgs/DynamixelCommand.h>
#include <dynamixel_workbench_msgs/DynamixelStateList.h>
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <cmath>
#include <vector>

enum MotorType {
    MX_28,
    AX_12A
};

/**
 * @brief Wrapper of goal position for a specific Dynamixel motor
 *
 */
struct DynamixelDriverGoal {
    MotorType type;
    int32_t goalPosition;
};

/**
 * @brief Controller of goal positions of the six Dynamixel motors
 *
 */
class DynamixelDriver {
private:
    std::vector<DynamixelDriverGoal> cmd;
    bool reachedNextGoal;

    void init();
    void setMsg();

public:
    // ROS instances
    ros::NodeHandle n_;
    ros::ServiceClient client_;

    // Messages published to RViz
    sensor_msgs::JointState visualization_msg;

    // Constructors
    DynamixelDriver();
    DynamixelDriver(std::vector<DynamixelDriverGoal>& goalList);
    DynamixelDriver(const uint8_t id, const int32_t goalpos);
    ~DynamixelDriver();

    // Command sending
    void send();

    // Command setting
    void set(const uint8_t id, const int32_t goalpos);
    void set(const std::vector<int>& goals);

    // Subscriber handler
    void handleState(const dynamixel_workbench_msgs::DynamixelStateList& msg);
    bool handleCommand(dynamixel_driver::MotorCommand::Request& req,
                       dynamixel_driver::MotorCommand::Response& res);
};

#endif