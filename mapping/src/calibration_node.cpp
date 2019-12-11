#include <ros/param.h>
#include <ros/ros.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <yaml-cpp/yaml.h>

#include <fstream>
#include <sstream>
#include <string>

namespace {

constexpr size_t num_cameras = 3;

enum class State {
    WaitingForCamera,
    Calibration,
    Finished
};

State state = State::WaitingForCamera;

struct Extrinsics {
    std::array<geometry_msgs::Transform, num_cameras> poses;

    Extrinsics() {
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

    bool calibrated(size_t i) {
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

void empty_callback(const std_msgs::Empty::Ptr &msg) {
}

}  // namespace

int main(int argc, char **argv) {
    ros::init(argc, argv, "Mapping");
    ros::NodeHandle node;
    ros::Rate loop_rate(60.0);

    // Listen and publish the extrinsics
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener listener(tf_buffer);
    static tf2_ros::StaticTransformBroadcaster broadcaster;

    Extrinsics extrinsics;

    ros::Time start_time(0);
    ros::Subscriber heartbeat_sub = node.subscribe("/calibration_heartbeat", 10, &empty_callback);

    size_t count = 0;
    while (ros::ok()) {
        switch (state) {
            case State::WaitingForCamera: {
                ROS_INFO("Waiting for calibration heartbeat");
                ros::spinOnce();
                if (heartbeat_sub.getNumPublishers() > 0) {
                    ROS_INFO("Aquired calibration heartbeat");
                    ROS_INFO("Waiting for pointclouds to settle.");

                    // Wait for pointclouds to settle
                    ros::Duration(5.0).sleep();

                    // Switch state
                    ROS_INFO("Switching to state CALIBRATION");
                    state = State::Calibration;

                    start_time = ros::Time::now();
                } else {
                    ros::Duration(1.0).sleep();
                }
                break;
            }
            case State::Calibration: {
                bool calibrated = true;
                for (size_t i = 0; i < num_cameras; i++) {
                    if (!extrinsics.calibrated(i)) {
                        try {
                            const auto pose_stamped = tf_buffer.lookupTransform("arm_bundle", "camera" + std::to_string(i + 1) + "_link", ros::Time(0));
                            extrinsics.poses[i] = pose_stamped.transform;

                            ROS_INFO("Camera %d calibrated", static_cast<int>(i + 1));
                        } catch (tf2::TransformException &ex) {
                            calibrated = false;
                            ROS_WARN("%s", ex.what());
                            ros::Duration(0.1).sleep();
                            continue;
                        }
                    }
                }

                if (calibrated) {
                    std::string file_name;
                    if (!node.getParam("extrinsics_file", file_name)) {
                        ROS_ERROR("Could not get extrinsics file from parameter server");
                    }
                    ROS_INFO("Saving extrinsics to %s", file_name.c_str());
                    YAML::Emitter out;

                    out << YAML::BeginMap;
                    for (size_t i = 0; i < num_cameras; i++) {
                        out << YAML::Key << "camera" + std::to_string(i + 1);
                        out << YAML::Value << YAML::BeginMap;

                        out << YAML::Key << "qw" << YAML::Value << extrinsics.poses[i].rotation.w;
                        out << YAML::Key << "qx" << YAML::Value << extrinsics.poses[i].rotation.x;
                        out << YAML::Key << "qy" << YAML::Value << extrinsics.poses[i].rotation.y;
                        out << YAML::Key << "qz" << YAML::Value << extrinsics.poses[i].rotation.z;
                        out << YAML::Key << "tx" << YAML::Value << extrinsics.poses[i].translation.x;
                        out << YAML::Key << "ty" << YAML::Value << extrinsics.poses[i].translation.y;
                        out << YAML::Key << "tz" << YAML::Value << extrinsics.poses[i].translation.z;

                        out << YAML::EndMap;
                    }
                    out << YAML::EndMap;

                    std::ofstream fout(file_name);
                    fout << out.c_str() << std::endl;
                    fout.close();

                    ROS_INFO("Saved extrinsics");
                    state = State::Finished;
                }

                break;
            }
            case State::Finished: {
                ROS_INFO_ONCE("Calibration finished. You may exit now.");
                ros::Duration(1.0).sleep();
                break;
            }
        }

        count++;
    }
}