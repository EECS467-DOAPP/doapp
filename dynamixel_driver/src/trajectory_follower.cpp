#include <dynamixel_driver/trajectory_follower.hpp>
#include <vector>

namespace doapp
{

TrajectoryFollower::TrajectoryFollower() : port_("/dev/ttyUSB0"), protocol_(2.0), baud_rate_(2000000)
{
    joints_.joint_ids = {0, 1, 2, 3, 4};
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

    const char **log;
    wb_.addSyncWriteHandler(30, 2);
    // for (size_t i = 0; i < Joints::num_joints; i++)
    // {
    //     if (!wb_.addSyncWriteHandler(joints_.joint_ids[i], "Goal_Position", log))
    //     {
    //         ROS_ERROR("Couldn't add sync write handler to id %d", joints_.joint_ids[i]);
    //         ROS_ERROR("%s", log[0]);
    //     }

    //     if (!wb_.addSyncReadHandler(joints_.joint_ids[i], "Present_Position", log))
    //     {
    //         ROS_ERROR("Couldn't add sync read handler to id %d", joints_.joint_ids[i]);
    //         ROS_ERROR("%s", log[0]);
    //     }
    // }
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
    std::vector<double> current_goal;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        current_goal = {0.0, 0.0, 0.0, 0.0, 0.0};
    }

    // static int16_t value = 500;
    // static int16_t direction = 1;
    // int32_t *data = reinterpret_cast<int32_t *>(current_goal_int.data());

    // current_goal_int.reserve(current_goal.size());
    // for (double g : current_goal)
    // {
    //     current_goal_int.push_back(wb_.convertRadian2Value(g, 1023, 0, M_PI / 2.0, -M_PI / 2.0));
    // }

    std::vector<int32_t> current_goal_int = {3500, 2000, 2000, 500, 500};
    wb_.syncWrite(0, joints_.joint_ids.data(), joints_.joint_ids.size(), current_goal_int.data(), 1);

    // for (auto id : joints_.joint_ids)
    // {
    //     wb_.goalPosition(id, (float)current_goal[id]);
    // }
    // wb_.syncWrite(0, joints_.joint_ids.data(), Joints::num_joints, data, 2);

    // Bulk write goal positions
    // Bulk read joint positions
    // for (size_t i = 0; i < Joints::num_joints; i++)
    // {
    //     int32_t goal_value = wb_.convertRadian2Value(joints_.joint_ids[i], 1023, 0, M_PI / 2.0f, -M_PI / 2.0f);
    //     if (!wb_.goalPosition(joints_.joint_ids[i], goal_value))
    //     {
    //         ROS_ERROR("Could not set goal poisiton");
    //     }

    //     wb_.setPositionControlMode()
    //     // if (!wb_.getPresentPositionData(joints_.joint_ids[i], ))
    // }

    // if (!wb_.initBulkWrite())
    // {
    //     ROS_ERROR("Could not initialize bulk write");
    // }
    // if (!wb_.initBulkRead())
    // {
    //     ROS_ERROR("Could not initialize bulk read");
    // }

    // for (size_t i = 0; i < 3; i++)
    // {
    //     if (!wb_.addBulkReadParam(joints_.joint_ids[i], 36, 2))
    //     {
    //         ROS_ERROR("Could not set bulk read param on joint %d", (int)i);
    //     }

    //     int32_t goal_value = wb_.convertRadian2Value(joints_.joint_ids[i], 1023, 0, M_PI / 2.0f, -M_PI / 2.0f);
    //     if (!wb_.addBulkWriteParam(joints_.joint_ids[i], 30, 2, goal_value))
    //     {
    //         ROS_ERROR("Could not set bulk write param on joint %d", (int)i);
    //     }
    // }

    // if (!wb_.bulkRead())
    // {
    //     ROS_ERROR("Could not bulk read");
    // }
    // if (!wb_.bulkWrite())
    // {
    //     ROS_ERROR("Could not bulk write");
    // }

    // int32_t *data;
    // wb_.getBulkReadData(data);

    // float *positions = reinterpret_cast<float *>(data);

    sensor_msgs::JointState state;
    state.header.stamp = ros::Time::now();
    state.position = current_goal;
    // std::copy(positions, positions + Joints::num_joints, std::back_inserter(state.position));

    return state;
}

} // namespace doapp
