#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Float64.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <dynamixel_workbench_toolbox/dynamixel_workbench.h>

#include <mutex>
#include <string>
#include <vector>

namespace doapp
{

struct Joints
{
    constexpr static size_t num_joints = 5;
    std::vector<uint8_t> joint_ids;
};

class TrajectoryFollower
{
public:
    TrajectoryFollower();
    ~TrajectoryFollower();

    void set_port(const std::string &port);
    void set_protocol(double protocol);
    void set_baud_rate(int baud_rate);

    void initialize_dynamixel();

    void trajectory_callback(const trajectory_msgs::JointTrajectoryConstPtr &trajectory_msg);

    void gripper_callback(const std_msgs::Float64ConstPtr &trajectory_msg);

    sensor_msgs::JointState follow_trajectory();

private:
    std::string port_;
    double protocol_;
    int baud_rate_;
    Joints joints_;
    DynamixelWorkbench wb_;

    std::mutex mtx_;
    trajectory_msgs::JointTrajectory current_trajectory_;
    double gripper_;
};
} // namespace doapp
