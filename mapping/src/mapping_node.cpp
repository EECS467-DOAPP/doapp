#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <yaml-cpp/yaml.h>

#include <occupancy_grid/occupancy_grid.hpp>

#include <string>
#include <sstream>
#include <fstream>

namespace
{

constexpr size_t num_cameras = 3;

enum class State
{
    WaitingForCamera,
    Mapping,
};

State state = State::WaitingForCamera;

struct Extrinsics
{
    std::array<geometry_msgs::Transform, num_cameras> poses;

    Extrinsics()
    {
        geometry_msgs::Transform identity;
        identity.rotation.w = 1.0;
        identity.rotation.x = 0.0;
        identity.rotation.y = 0.0;
        identity.rotation.z = 0.0;

        identity.translation.x = 0.0;
        identity.translation.y = 0.0;
        identity.translation.z = 0.0;

        poses[0] = identity;
        poses[1] = identity;
        poses[2] = identity;
    }

    bool calibrated(size_t i)
    {
        geometry_msgs::Transform identity;
        identity.rotation.w = 1.0;
        identity.rotation.x = 0.0;
        identity.rotation.y = 0.0;
        identity.rotation.z = 0.0;

        identity.translation.x = 0.0;
        identity.translation.y = 0.0;
        identity.translation.z = 0.0;

        // The pose should not be the identity (The apriltags are not in the camera)
        return poses[i].rotation.w != identity.rotation.w ||
               poses[i].rotation.x != identity.rotation.x ||
               poses[i].rotation.y != identity.rotation.y ||
               poses[i].rotation.z != identity.rotation.z ||
               poses[i].translation.x != identity.translation.x ||
               poses[i].translation.y != identity.translation.y ||
               poses[i].translation.z != identity.translation.z;
    }
};

void empty_callback(const std_msgs::Empty::Ptr &msg)
{
}

void read_extrinsics(Extrinsics &extrinsics, YAML::Node node)
{
    for (size_t i = 0; i < num_cameras; i++)
    {
        extrinsics.poses[i].rotation.w = node["camera" + std::to_string(i + 1)]["qw"].as<double>();
        extrinsics.poses[i].rotation.x = node["camera" + std::to_string(i + 1)]["qx"].as<double>();
        extrinsics.poses[i].rotation.y = node["camera" + std::to_string(i + 1)]["qy"].as<double>();
        extrinsics.poses[i].rotation.z = node["camera" + std::to_string(i + 1)]["qz"].as<double>();
        extrinsics.poses[i].translation.x = node["camera" + std::to_string(i + 1)]["tx"].as<double>();
        extrinsics.poses[i].translation.y = node["camera" + std::to_string(i + 1)]["ty"].as<double>();
        extrinsics.poses[i].translation.z = node["camera" + std::to_string(i + 1)]["tz"].as<double>();
    }
}

} // namespace

int main(int argc, char **argv)
{
    ros::init(argc, argv, "Mapping");
    ros::NodeHandle node;
    ros::Rate loop_rate(60.0);

    // Listen and publish the extrinsics
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener listener(tf_buffer);
    static tf2_ros::StaticTransformBroadcaster broadcaster;
    static tf2_ros::TransformBroadcaster br;

    bool publish_tf = false;
    node.getParam("publish_pointclouds", publish_tf);

    // Read extrinsics from calibration file
    Extrinsics extrinsics;
    std::string file_name;
    if (!node.getParam("extrinsics_file", file_name))
    {
        ROS_ERROR("Could not get extrinsics file from parameter server");
    }
    ROS_INFO("Reading extrinsics from %s", file_name.c_str());
    YAML::Node calibration_node = YAML::LoadFile(file_name);
    read_extrinsics(extrinsics, calibration_node);

    ros::Time start_time(0);
    ros::Subscriber heartbeat_sub = node.subscribe("/mapping_heartbeat", 10, &empty_callback);

    // Publish static transforms
    const auto now = ros::Time::now();
    for (size_t i = 0; i < num_cameras; i++)
    {
        geometry_msgs::TransformStamped static_transform;
        static_transform.header.frame_id = "arm_bundle";
        static_transform.header.stamp = now;

        static_transform.transform = extrinsics.poses[i];

        std::string camera_frame = "camera" + std::to_string(i + 1) + "_link";
        static_transform.child_frame_id = camera_frame;

        broadcaster.sendTransform(static_transform);
    }

    size_t count = 0;
    while (ros::ok())
    {
        switch (state)
        {
        case State::WaitingForCamera:
        {
            ROS_INFO("Waiting for heartbeat");
            ros::spinOnce();
            if (heartbeat_sub.getNumPublishers() > 0)
            {
                ROS_INFO("Aquired heartbeat");
                ROS_INFO("Waiting for cameras to settle.");

                // Wait for cameras to settle
                ros::Duration(5.0).sleep();

                // Switch state
                ROS_INFO("Switching to state Mapping");
                state = State::Mapping;

                start_time = ros::Time::now();
            }
            else
            {
                ros::Duration(1.0).sleep();
            }
            break;
        }
        case State::Mapping:
        {
            ROS_INFO("Mapping");
            break;
        }
        }

        count++;
    }
}