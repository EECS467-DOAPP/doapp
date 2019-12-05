#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/static_transform_broadcaster.h>

namespace
{

enum class State
{
    Calibration,
    Mapping,
};

struct Extrinsics
{
    geometry_msgs::TransformStamped camera1_pose;
    geometry_msgs::TransformStamped camera2_pose;
    geometry_msgs::TransformStamped camera3_pose;
};

} // namespace

int main(int argc, char **argv)
{
    ros::init(argc, argv, "Mapping");
    ros::NodeHandle node;
    ros::Rate loop_rate(100.0);

    const auto start_time = ros::Time::now();

    // Listen and publish the extrinsics
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener listener(tf_buffer);
    static tf2_ros::StaticTransformBroadcaster broadcaster;

    Extrinsics extrinsics;

    State state = State::Calibration;

    // Wait for pointclouds to settle
    ros::Duration(5.0).sleep();

    size_t count = 0;
    while (ros::ok())
    {
        switch (state)
        {
        case State::Calibration:
        {
            const auto uptime = ros::Time::now() - start_time;
            const auto calibration_duration = ros::Duration(10.0);
            if (uptime > calibration_duration)
            {
                ROS_ERROR("Couldn't calibrate in time.");
                ros::shutdown();
            }

            try
            {
                extrinsics.camera1_pose = tf_buffer.lookupTransform("arm_base", "camera1_color_optical_frame", ros::Time(0));
                extrinsics.camera2_pose = tf_buffer.lookupTransform("arm_base", "camera2_color_optical_frame", ros::Time(0));
                extrinsics.camera3_pose = tf_buffer.lookupTransform("arm_base", "camera3_color_optical_frame", ros::Time(0));

                ROS_INFO("Calibrated camera extrinsics");

                // Change state once calibrated
                state = State::Mapping;
            }
            catch (tf2::TransformException &ex)
            {
                ROS_WARN("%s", ex.what());
                ros::Duration(1.0).sleep();
                continue;
            }
            break;
        }
        case State::Mapping:
        {

            break;
        }
        }

        count++;
    }
}