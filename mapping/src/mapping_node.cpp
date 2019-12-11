#include <ros/ros.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2/transform_datatypes.h>
#include <tf2/convert.h>
#include <tf2/LinearMath/Transform.h>
#include <yaml-cpp/yaml.h>

#include <image_geometry/pinhole_camera_model.h>
#include <image_transport/image_transport.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <depth_image_proc/depth_conversions.h>

#include <std_msgs/Empty.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>

#include <occupancy_grid/occupancy_grid.hpp>

#include <string>
#include <fstream>
#include <mutex>
#include <functional>

namespace
{

using PointCloud = pcl::PointCloud<pcl::PointXYZ>;
constexpr size_t num_cameras = 3;

enum class State
{
    WaitingForCamera,
    Mapping,
};

State state = State::WaitingForCamera;

struct Extrinsics
{
    std::array<geometry_msgs::TransformStamped, num_cameras> poses_msg;

    Extrinsics()
    {
    }

}; // namespace

Extrinsics extrinsics;

void read_extrinsics(Extrinsics &extrinsics, YAML::Node node)
{
    for (size_t i = 0; i < num_cameras; i++)
    {
        geometry_msgs::TransformStamped transform;
        transform.child_frame_id = "camera" + std::to_string(i + 1) + "_link";
        transform.header.frame_id = "arm_bundle";
        transform.header.stamp = ros::Time::now();

        transform.transform.rotation.w = (node["camera" + std::to_string(i + 1)]["qw"].as<double>());
        transform.transform.rotation.x = (node["camera" + std::to_string(i + 1)]["qx"].as<double>());
        transform.transform.rotation.y = (node["camera" + std::to_string(i + 1)]["qy"].as<double>());
        transform.transform.rotation.z = (node["camera" + std::to_string(i + 1)]["qz"].as<double>());
        transform.transform.translation.x = (node["camera" + std::to_string(i + 1)]["tx"].as<double>());
        transform.transform.translation.y = (node["camera" + std::to_string(i + 1)]["ty"].as<double>());
        transform.transform.translation.z = (node["camera" + std::to_string(i + 1)]["tz"].as<double>());

        extrinsics.poses_msg[i] = transform;
    }
}

void empty_callback(const std_msgs::Empty::Ptr &msg)
{
}

// Handles float or uint16 depths
void convert_sensor_msg_to_pcl(
    const sensor_msgs::ImageConstPtr &depth_msg,
    PointCloud::Ptr &cloud_msg,
    const image_geometry::PinholeCameraModel &model,
    double range_max = 0.0)
{
    // Use correct principal point from calibration
    float center_x = model.cx();
    float center_y = model.cy();

    // Combine unit conversion (if necessary) with scaling by focal length for computing (X,Y)
    double unit_scaling = depth_image_proc::DepthTraits<uint16_t>::toMeters(1);
    float constant_x = unit_scaling / model.fx();
    float constant_y = unit_scaling / model.fy();
    float bad_point = std::numeric_limits<float>::quiet_NaN();

    cloud_msg->reserve(depth_msg->width * depth_msg->height);

    constexpr size_t num_threads = 4;

    const uint16_t *depth_row = reinterpret_cast<const uint16_t *>(&depth_msg->data[0]);
    int row_step = depth_msg->step / sizeof(uint16_t);
    for (int v = 0; v < (int)depth_msg->height; v++, depth_row += row_step)
    {
        for (int u = 0; u < (int)depth_msg->width; u++)
        {
            uint16_t depth = depth_row[u];

            // Missing points denoted by NaNs
            if (!depth_image_proc::DepthTraits<uint16_t>::valid(depth))
            {
                if (range_max != 0.0)
                {
                    depth = depth_image_proc::DepthTraits<uint16_t>::fromMeters(range_max);
                }
                else
                {
                    continue;
                }
            }

            // Fill in XYZ
            const float x = (u - center_x) * depth * constant_x;
            const float y = (v - center_y) * depth * constant_y;
            const float z = depth_image_proc::DepthTraits<uint16_t>::toMeters(depth);
            cloud_msg->push_back(pcl::PointXYZ(x, y, z));
        }
    }
}

double size = 1.0;
double granularity = 0.01;
// Radius outlier removal
double radius = 0.05;
int min_neightbors_in_radius = 50;

PointCloud cloud;
tf2_ros::Buffer tf_buffer;

std::mutex merge_mtx;
std::mutex camera1_mtx;
std::mutex camera2_mtx;
std::mutex camera3_mtx;
bool camera1 = false;
bool camera2 = false;
bool camera3 = false;
PointCloud::Ptr camera1_cloud_pcl(new PointCloud), camera2_cloud_pcl(new PointCloud), camera3_cloud_pcl(new PointCloud);

pcl::CropBox<pcl::PointXYZ> box_filter_camera1;
pcl::CropBox<pcl::PointXYZ> box_filter_camera2;
pcl::CropBox<pcl::PointXYZ> box_filter_camera3;
pcl::VoxelGrid<pcl::PointXYZ> voxel_filter_camera1;
pcl::VoxelGrid<pcl::PointXYZ> voxel_filter_camera2;
pcl::VoxelGrid<pcl::PointXYZ> voxel_filter_camera3;

bool new_cloud = false;

void depth_callback(int camera_id, const sensor_msgs::ImageConstPtr &depth_msg, const sensor_msgs::CameraInfoConstPtr &info_msg)
{
    // Deproject points into a point cloud
    image_geometry::PinholeCameraModel model;
    model.fromCameraInfo(info_msg);

    PointCloud::Ptr cloud_msg(new PointCloud);
    PointCloud::Ptr cropped_cloud(new PointCloud);
    PointCloud::Ptr downsampled_cloud(new PointCloud);

    // Do deprojection
    cloud_msg->width = depth_msg->width;
    cloud_msg->height = depth_msg->height;
    cloud_msg->header = pcl_conversions::toPCL(depth_msg->header);
    cloud_msg->is_dense = false;
    convert_sensor_msg_to_pcl(depth_msg, cloud_msg, model);

    switch (camera_id)
    {
    case 1:
    {
        std::lock_guard<std::mutex> lock(camera1_mtx);
        if (!camera1)
        {
            voxel_filter_camera1.setInputCloud(cloud_msg);
            voxel_filter_camera1.filter(*downsampled_cloud);
            box_filter_camera1.setInputCloud(downsampled_cloud);
            box_filter_camera1.filter(*cropped_cloud);

            pcl_ros::transformPointCloud("arm_bundle", *cropped_cloud, *camera1_cloud_pcl, tf_buffer);
            camera1_cloud_pcl->header = pcl_conversions::toPCL(depth_msg->header);

            camera1 = true;
        }
        break;
    }
    case 2:
    {
        std::lock_guard<std::mutex> lock(camera2_mtx);
        if (!camera2)
        {
            voxel_filter_camera2.setInputCloud(cloud_msg);
            voxel_filter_camera2.filter(*downsampled_cloud);
            box_filter_camera2.setInputCloud(downsampled_cloud);
            box_filter_camera2.filter(*cropped_cloud);

            pcl_ros::transformPointCloud("arm_bundle", *cropped_cloud, *camera2_cloud_pcl, tf_buffer);
            camera2_cloud_pcl->header = pcl_conversions::toPCL(depth_msg->header);
            camera2 = true;
        }
        break;
    }
    case 3:
    {
        std::lock_guard<std::mutex> lock(camera3_mtx);
        if (!camera3)
        {
            voxel_filter_camera3.setInputCloud(cloud_msg);
            voxel_filter_camera3.filter(*downsampled_cloud);
            box_filter_camera3.setInputCloud(downsampled_cloud);
            box_filter_camera3.filter(*cropped_cloud);

            pcl_ros::transformPointCloud("arm_bundle", *cropped_cloud, *camera3_cloud_pcl, tf_buffer);
            camera3_cloud_pcl->header = pcl_conversions::toPCL(depth_msg->header);

            camera3 = true;
        }
        break;
    }
    }

    {
        std::lock_guard<std::mutex> lock1(camera1_mtx);
        std::lock_guard<std::mutex> lock2(camera2_mtx);
        std::lock_guard<std::mutex> lock3(camera3_mtx);
        if (camera1 && camera2 && camera3)
        {
            {
                std::lock_guard<std::mutex> lock(merge_mtx);
                PointCloud::Ptr merged_cloud(new PointCloud);
                cloud = *camera1_cloud_pcl;
                cloud += *camera2_cloud_pcl;
                cloud += *camera3_cloud_pcl;

                cloud.header = camera1_cloud_pcl->header;
                new_cloud = true;
            }
            camera1 = false;
            camera2 = false;
            camera3 = false;
        }
    }
}

void set_filter_parameters()
{
    box_filter_camera1.setMin(Eigen::Vector4f(-size / 2.0, -size / 2.0, 0.0, 1.0));
    box_filter_camera1.setMax(Eigen::Vector4f(size / 2.0, size / 2.0, size, 1.0));
    box_filter_camera2.setMin(Eigen::Vector4f(-size / 2.0, -size / 2.0, 0.0, 1.0));
    box_filter_camera2.setMax(Eigen::Vector4f(size / 2.0, size / 2.0, size, 1.0));
    box_filter_camera3.setMin(Eigen::Vector4f(-size / 2.0, -size / 2.0, 0.0, 1.0));
    box_filter_camera3.setMax(Eigen::Vector4f(size / 2.0, size / 2.0, size, 1.0));

    Eigen::Isometry3d camera1_isom = tf2::transformToEigen(extrinsics.poses_msg[0]).inverse();
    Eigen::Isometry3d camera2_isom = tf2::transformToEigen(extrinsics.poses_msg[1]).inverse();
    Eigen::Isometry3d camera3_isom = tf2::transformToEigen(extrinsics.poses_msg[2]).inverse();

    Eigen::Affine3f camera1_affine(camera1_isom.affine().cast<float>());
    Eigen::Affine3f camera2_affine(camera2_isom.affine().cast<float>());
    Eigen::Affine3f camera3_affine(camera3_isom.affine().cast<float>());

    box_filter_camera1.setTransform(camera1_affine);
    box_filter_camera2.setTransform(camera2_affine);
    box_filter_camera3.setTransform(camera3_affine);

    voxel_filter_camera1.setLeafSize(granularity, granularity, granularity);
    voxel_filter_camera2.setLeafSize(granularity, granularity, granularity);
    voxel_filter_camera3.setLeafSize(granularity, granularity, granularity);
}

} // namespace

int main(int argc, char **argv)
{
    ros::init(argc, argv, "Mapping");
    ros::NodeHandle node;
    ros::Rate loop_rate(100.0);

    node.getParam("size", size);
    node.getParam("granularity", granularity);
    node.getParam("radius", radius);
    node.getParam("min_neightbors_in_radius", min_neightbors_in_radius);

    ROS_INFO("Mapping with size %f meters and a granularity of %f meters", size, granularity);

    // Listen and publish the extrinsics
    tf2_ros::TransformListener listener(tf_buffer);
    static tf2_ros::StaticTransformBroadcaster broadcaster;

    bool publish_tf = false;
    node.getParam("publish_pointclouds", publish_tf);

    // Read extrinsics from calibration file
    std::string file_name;
    if (!node.getParam("extrinsics_file", file_name))
    {
        ROS_ERROR("Could not get extrinsics file from parameter server");
    }
    ROS_INFO("Reading extrinsics from %s", file_name.c_str());
    YAML::Node calibration_node = YAML::LoadFile(file_name);
    read_extrinsics(extrinsics, calibration_node);

    set_filter_parameters();

    ros::Time start_time(0);
    ros::Subscriber heartbeat_sub = node.subscribe("/mapping_heartbeat", 10, &empty_callback);

    image_transport::ImageTransport camera1_transport(node);
    image_transport::ImageTransport camera2_transport(node);
    image_transport::ImageTransport camera3_transport(node);

    image_transport::CameraSubscriber camera1_sub;
    image_transport::CameraSubscriber camera2_sub;
    image_transport::CameraSubscriber camera3_sub;

    {
        namespace ph = std::placeholders;
        // image_transport::TransportHints hints("raw", ros::TransportHints(), node);
        camera1_sub = camera1_transport.subscribeCamera("/camera1/depth/image_rect_raw", 1, std::bind(&depth_callback, 1, ph::_1, ph::_2));
        camera2_sub = camera2_transport.subscribeCamera("/camera2/depth/image_rect_raw", 1, std::bind(&depth_callback, 2, ph::_1, ph::_2));
        camera3_sub = camera3_transport.subscribeCamera("/camera3/depth/image_rect_raw", 1, std::bind(&depth_callback, 3, ph::_1, ph::_2));
    }

    ros::Publisher cloud_pub = node.advertise<PointCloud>("/merged_cloud", 1);
    ros::Publisher empty_pub = node.advertise<std_msgs::Empty>("/mapping_empty", 1);

    // Publish static transforms
    for (size_t i = 0; i < num_cameras; i++)
    {
        broadcaster.sendTransform(extrinsics.poses_msg[i]);
    }

    doapp::OccupancyGrid grid(size, granularity);

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
                ROS_INFO("Waiting for extrinsics to settle.");

                // Wait for extrinsics to settle
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
            ros::spinOnce();

            {
                std::lock_guard<std::mutex> lock(merge_mtx);
                if (new_cloud)
                {
                    cloud.header.frame_id = "arm_bundle";

                    int num_points = cloud.size();
                    ROS_INFO("There are %d points in the cloud.", num_points);
                    std_msgs::Empty empty_msg;
                    empty_pub.publish(empty_msg);
                    new_cloud = false;

                    cloud_pub.publish(cloud);
                }
            }

            loop_rate.sleep();
            break;
        }
        }

        count++;
    }
}