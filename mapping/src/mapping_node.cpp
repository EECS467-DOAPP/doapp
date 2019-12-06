#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/static_transform_broadcaster.h>

#include <string>
#include <sstream>
#include <mutex>

namespace
{

constexpr size_t num_cameras = 3;

enum class State
{
    WaitingForCamera,
    Calibration,
    WaitingForCalibrationEnd,
    WaitingForReboot,
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

std::mutex mtx;
ros::Time last_empty(0);

void empty_callback(const std_msgs::Empty::Ptr &msg)
{
    std::lock_guard<std::mutex> lock(mtx);
    last_empty = ros::Time::now();
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

    Extrinsics extrinsics;

    ros::Time start_time(0);
    ros::Subscriber calibration_heartbeat_sub = node.subscribe("/calibration_heartbeat", 10, &empty_callback);
    ros::Subscriber mapping_heartbeat_sub = node.subscribe("/mapping_heartbeat", 10, &empty_callback);

    size_t count = 0;
    while (ros::ok())
    {
        switch (state)
        {
        case State::WaitingForCamera:
        {
            ROS_INFO("Waiting for calibration heartbeat");
            ros::spinOnce();
            if (calibration_heartbeat_sub.getNumPublishers() > 0)
            {
                ROS_INFO("Aquired calibration heartbeat");
                ROS_INFO("Waiting for pointclouds to settle.");

                // Wait for pointclouds to settle
                ros::Duration(5.0).sleep();

                // Switch state
                ROS_INFO("Switching to state CALIBRATION");
                state = State::Calibration;

                start_time = ros::Time::now();
            }
            else
            {
                ros::Duration(1.0).sleep();
            }
            break;
        }
        case State::Calibration:
        {
            ros::spinOnce();
            const auto uptime = ros::Time::now() - start_time;
            const auto calibration_duration = ros::Duration(30.0);
            if (uptime > calibration_duration)
            {
                ROS_ERROR("CALIBRATION FAILED! Couldn't calibrate in time.");
                ros::shutdown();
            }

            bool calibrated = true;
            for (size_t i = 0; i < num_cameras; i++)
            {
                if (!extrinsics.calibrated(i))
                {
                    try
                    {
                        std::stringstream ss;
                        ss << "camera" << i + 1 << "_color_optical_frame";
                        std::string camera_frame = ss.str();

                        const auto pose_stamped = tf_buffer.lookupTransform("arm_bundle", camera_frame, ros::Time(0));
                        extrinsics.poses[i] = pose_stamped.transform;

                        ROS_INFO("Camera %d calibrated", i + 1);
                    }
                    catch (tf2::TransformException &ex)
                    {
                        calibrated = false;
                        ROS_WARN("%s", ex.what());
                        ros::Duration(0.1).sleep();
                        continue;
                    }
                }
            }

            if (calibrated)
            {
                for (size_t i = 0; i < num_cameras; i++)
                {

                    node.setParam("/calibration/camera" + std::to_string(i + 1) + "/rotation/w", std::to_string(extrinsics.poses[i].rotation.w));
                    node.setParam("/calibration/camera" + std::to_string(i + 1) + "/rotation/x", std::to_string(extrinsics.poses[i].rotation.x));
                    node.setParam("/calibration/camera" + std::to_string(i + 1) + "/rotation/y", std::to_string(extrinsics.poses[i].rotation.y));
                    node.setParam("/calibration/camera" + std::to_string(i + 1) + "/rotation/z", std::to_string(extrinsics.poses[i].rotation.z));

                    node.setParam("/calibration/camera" + std::to_string(i + 1) + "/translation/x", std::to_string(extrinsics.poses[i].translation.x));
                    node.setParam("/calibration/camera" + std::to_string(i + 1) + "/translation/y", std::to_string(extrinsics.poses[i].translation.y));
                    node.setParam("/calibration/camera" + std::to_string(i + 1) + "/translation/z", std::to_string(extrinsics.poses[i].translation.z));
                }
            }

            ROS_INFO("Set extrinsics to parameter server");
            ros::shutdown();

            break;
        }
        case State::WaitingForCalibrationEnd:
        {
            // Wait for apriltag nodes to end
            ros::spinOnce();
            ROS_INFO_ONCE("Waiting for calibration to end");
            ros::Time last_empty_copy;
            {
                std::lock_guard<std::mutex> lock(mtx);
                last_empty_copy = last_empty;
            }

            if (ros::Time::now() - last_empty_copy >= ros::Duration(1.0))
            {
                ROS_INFO_ONCE("Calibration shutdown");
                calibration_heartbeat_sub.shutdown();
                state = State::WaitingForReboot;
            }
            else
            {
                ros::Duration(1.0).sleep();
            }
            break;
        }
        case State::WaitingForReboot:
        {

            // Wait for cameras to launch again
            ROS_INFO("Waiting for mapping heartbeat");
            ros::spinOnce();
            if (mapping_heartbeat_sub.getNumPublishers() > 0)
            {
                ROS_INFO("Aquired mapping heartbeat");

                // Publish static transform
                ROS_INFO("Publishing static transforms");
                for (size_t i = 0; i < num_cameras; i++)
                {
                    geometry_msgs::TransformStamped static_transform;
                    static_transform.transform = extrinsics.poses[i];
                    static_transform.header.stamp = ros::Time::now();
                    static_transform.header.frame_id = "arm_bundle";

                    std::stringstream ss;
                    ss << "camera" << i + 1 << "_color_optical_frame";
                    std::string camera_frame = ss.str();
                    static_transform.child_frame_id = camera_frame;
                }

                // Change state to mapping
                state = State::Mapping;

                mapping_heartbeat_sub.shutdown();
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
            // Publish static transforms
            break;
        }
        }

        count++;
    }
}