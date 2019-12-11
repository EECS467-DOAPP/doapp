#include <dynamixel_driver/trajectory_follower.hpp>
#include <vector>
#include <iostream>

constexpr double DEG2RAD = (2*M_PI/360.0);
namespace doapp
{

TrajectoryFollower::TrajectoryFollower() : port_("/dev/ttyUSB0"), protocol_(2.0), baud_rate_(2000000), prev_waypt_(0), next_waypt_(0)
{
    current_trajectory_.header.stamp = ros::Time::now();
    trajectory_msgs::JointTrajectoryPoint point;

    point.positions = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    point.velocities = {1.5, 1.5, 1.5, 1.5, 1.5, 1.5};
    point.time_from_start = ros::Duration(0.0);
    current_trajectory_.points.push_back(point);


    point.positions = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    point.velocities = {1.5, 1.5, 1.5, 1.5, 1.5, 1.5};
    point.time_from_start = ros::Duration(2.0);
    current_trajectory_.points.push_back(point);

    point.positions = {0.0, 1.0, 0.0, 0.5, 0.0, 0.0};
    point.velocities = {1.5, 1.5, 1.5, 1.5, 1.5, 1.5};
    point.time_from_start = ros::Duration(3.0);
    current_trajectory_.points.push_back(point);

    point.positions = {117.14 * DEG2RAD, 25.78 * DEG2RAD, 52.26 * DEG2RAD, 70.79 * DEG2RAD, 0, 0};
    point.velocities = {1.5, 1.5, 1.5, 1.5, 1.5, 1.5};
    //point.velocities = {0, 0, 0, 0, 0, 0};

    //point.positions = {1.0, 1.0, 0.0, 0.5, 1.0, 0.0};
    //point.velocities = {1.5, 1.5, 1.5, 1.5, 1.5, 1.5};
    point.time_from_start = ros::Duration(4.0);
    current_trajectory_.points.push_back(point);

    point.positions = {0.0, 0.0, 1.0, -0.5, 1.0, 0.75};
    point.velocities = {1.5, 1.5, 1.5, 1.5, 1.5, 1.5};
    point.time_from_start = ros::Duration(5.0);
    current_trajectory_.points.push_back(point);

    point.positions = {0.0, -1.0, 1.5, -1.5, 1.0, 0.75};
    point.velocities = {1.5, 1.5, 1.5, 1.5, 1.5, 1.5};
    point.time_from_start = ros::Duration(6.0);
    current_trajectory_.points.push_back(point);

    point.positions = {0.0, -0.0, 0.5, -0.5, 0.0, 0.0};
    point.velocities = {1.5, 1.5, 1.5, 1.5, 1.5, 1.5};
    point.time_from_start = ros::Duration(7.0);
    current_trajectory_.points.push_back(point);

    point.positions = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    point.velocities = {1.5, 1.5, 1.5, 1.5, 1.5, 1.5};
    point.time_from_start = ros::Duration(8.0);
    current_trajectory_.points.push_back(point);
}

TrajectoryFollower::~TrajectoryFollower()
{
}

void TrajectoryFollower::set_port(const std::string &port)
{
    port_ = port;
}

void TrajectoryFollower::set_protocol(double protocol)
{
    protocol_ = protocol;
}

void TrajectoryFollower::set_baud_rate(int baud_rate)
{
    baud_rate_ = baud_rate;
}

void TrajectoryFollower::initialize_dynamixel()
{
    if (!wb_.init(port_.c_str(), baud_rate_))
    {
        ROS_ERROR("Could not initialize dynamixel at port %s with baud rate %d", port_.c_str(), baud_rate_);
        ros::shutdown();
    }
    ROS_INFO("Succeeded to initialize port %s with baud rate %d", port_.c_str(), baud_rate_);

    for (auto id : joints_.joint_ids)
    {
        float version = wb_.getProtocolVersion();
        ROS_INFO("Joint %d is using protocol version %f", id, version);
    }

    for (auto id : joints_.joint_ids)
    {
        if (!wb_.ping(id))
        {
            ROS_ERROR("Could not ping joint %d", id);
            ros::shutdown();
        }
    }
    ROS_INFO("Successfully pinged all joints");

    for (auto id : joints_.joint_ids)
    {
        if (!wb_.torqueOn(id))
        {
            ROS_ERROR("Could not enable torque on joint %d", id);
            ros::shutdown();
        }
    }
    ROS_INFO("Enabled torque on joints");

    for (auto id : joints_.joint_ids)
    {
        if (!wb_.jointMode(id))
        {
            ROS_ERROR("Could not set joint mode on joint %d", id);
            ros::shutdown();
        }
    }
    ROS_INFO("Set joint mode");

    constexpr uint16_t goal_position_address = 30;
    constexpr uint16_t present_position_address = 36;

    // Writes 4 bytes (2 to goal position and 2 to goal velocity)
    if (!wb_.addSyncWriteHandler(goal_position_address, 4))
    {
        ROS_ERROR("Failed to add a sync write handler");
        ros::shutdown();
    }
    // Reads 2 bytes from present position
    if (!wb_.addSyncReadHandler(present_position_address, 6))
    {
        ROS_ERROR("Failed to add a sync read handler");
        ros::shutdown();
    }
    ROS_INFO("Set sync handlers");
}

void TrajectoryFollower::trajectory_callback(const trajectory_msgs::JointTrajectoryConstPtr &trajectory_msg)
{
    std::lock_guard<std::mutex> lock(mtx_);
    current_trajectory_ = *trajectory_msg;
    std::cout << "NEW TRAJECTORY WITH " << current_trajectory_.points.size() << " POINTS" << std::endl;
    prev_waypt_ = 0;
    if (current_trajectory_.points.size() > 1)
    {
        next_waypt_ = 1;
    }
    else
    {
        next_waypt_ = 0;
    }
}

void TrajectoryFollower::gripper_callback(const std_msgs::Float64ConstPtr &trajectory_msg)
{
    std::lock_guard<std::mutex> lock(mtx_);
    gripper_ = trajectory_msg->data;
}

sensor_msgs::JointState TrajectoryFollower::follow_trajectory()
{
    sensor_msgs::JointState state;
    auto now = ros::Time::now();
    state.header.stamp = now;

    const auto traj_start = current_trajectory_.header.stamp;
    const auto prev_time = traj_start + current_trajectory_.points[prev_waypt_].time_from_start;
    const auto next_time = traj_start + current_trajectory_.points[prev_waypt_].time_from_start;
    const auto segment_duration = next_time - prev_time;

    if (prev_waypt_ == next_waypt_)
    {
        state.position = current_trajectory_.points.back().positions;
    }
    else
    {
        const auto current_duration = now - prev_time;

        const auto segment_progress = std::min(1.0, current_duration.toSec() / segment_duration.toSec());
        state.position.reserve(joints_.num_joints);

        const auto &prev_point = current_trajectory_.points[prev_waypt_];
        const auto &next_point = current_trajectory_.points[next_waypt_];
        for (size_t i = 0; i < joints_.num_joints; i++)
        {
            state.position[i] = (1.0 - segment_progress) * prev_point.positions[i] + segment_progress * next_point.positions[i];
        }
    }

    // Get current goal from trajectory
    std::vector<double> goal_angle;
    std::vector<double> goal_velocity;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        goal_angle = current_trajectory_.points[prev_waypt_].positions;
        goal_velocity = current_trajectory_.points[prev_waypt_].velocities;
    }

    std::vector<int16_t> goal_data;
    goal_data.reserve(joints_.num_joints);

    for (size_t i = 0; i < joints_.num_joints; i++)
    {
        goal_data.push_back(wb_.convertRadian2Value(joints_.joint_ids[i], goal_angle[i]));
        goal_data.push_back(wb_.convertVelocity2Value(joints_.joint_ids[i], goal_velocity[i]));
    }

    int32_t *write_data = reinterpret_cast<int32_t *>(goal_data.data());
    if (!wb_.syncWrite(0, joints_.joint_ids.data(), joints_.num_joints, write_data, 1))
    {
        ROS_WARN("Could not sync write to dynamixel");
    }

    if (now > next_time)
    {
        prev_waypt_ = next_waypt_;
        if (next_waypt_ + 1 < current_trajectory_.points.size())
        {
            next_waypt_++;
        }
    }

    return state;
}

} // namespace doapp
