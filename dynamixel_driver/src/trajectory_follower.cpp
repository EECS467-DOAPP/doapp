#include <dynamixel_driver/trajectory_follower.hpp>
#include <vector>

namespace doapp
{

TrajectoryFollower::TrajectoryFollower() : port_("/dev/ttyUSB0"), protocol_(2.0), baud_rate_(2000000)
{
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

    if (!wb_.addSyncWriteHandler(30, 4))
    {
        ROS_ERROR("Failed to add a sync write handler");
    }
    if (!wb_.addSyncReadHandler(36, 2))
    {
        ROS_ERROR("Failed to add a sync read handler");
    }
    ROS_INFO("Set sync handlers");
}

void TrajectoryFollower::trajectory_callback(const trajectory_msgs::JointTrajectoryConstPtr &trajectory_msg)
{
    std::lock_guard<std::mutex> lock(mtx_);
    current_trajectory_ = *trajectory_msg;
}

void TrajectoryFollower::gripper_callback(const std_msgs::Float64ConstPtr &trajectory_msg)
{
    std::lock_guard<std::mutex> lock(mtx_);
    gripper_ = trajectory_msg->data;
}

sensor_msgs::JointState TrajectoryFollower::follow_trajectory()
{

    // Get current goal from trajectory
    std::vector<double> goal_angle;
    std::vector<double> goal_velocity;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        goal_angle = {0.0, 0.0, 0.0, 0.0, 0.0, gripper_};
        goal_velocity = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
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

    int32_t read_data[joints_.num_joints];
    if (!wb_.syncRead(0, joints_.joint_ids.data(), joints_.num_joints))
    {
        ROS_WARN("Could not sync read from dynamixel");
    }
    else
    {
        if (!wb_.getSyncReadData(0, read_data))
        {
            ROS_WARN("Could not retrieve data from sync read");
        }
    }

    // ROS_INFO("Read data");
    sensor_msgs::JointState state;
    state.header.stamp = ros::Time::now();
    // state.position.reserve(joints_.num_joints);
    // for (size_t i = 0; i < joints_.num_joints; i++)
    // {
    //     state.position.push_back(wb_.convertValue2Radian(joints_.joint_ids[i], read_data[i]));
    // }

    return state;
}

} // namespace doapp
