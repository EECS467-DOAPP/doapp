#include <dynamixel_driver/trajectory_follower.hpp>

namespace {

}  // namespace

int main(int argc, char **argv) {
    ros::init(argc, argv, "trajectory_follower");
    ros::NodeHandle nh;
    ros::Rate loop_rate(100.0);

    doapp::TrajectoryFollower follower;

    ros::Publisher joint_pub = nh.advertise<sensor_msgs::JointState>("/joint_state", 1);

    ros::Subscriber trajectory_sub = nh.subscribe("/joint_trajectory", 1, &doapp::TrajectoryFollower::trajectory_callback, &follower);
    ros::Subscriber gripper_sub = nh.subscribe("/gripper", 1, &doapp::TrajectoryFollower::gripper_callback, &follower);

    std::string port;
    double protocol;
    int baud_rate;

    nh.param("port", port, std::string("/dev/ttyUSB0"));
    nh.param("dynamixel_protocol", protocol, 2.0);
    nh.param("baud_rate", baud_rate, 2000000);

    follower.set_port(port);
    follower.set_protocol(protocol);
    follower.set_baud_rate(baud_rate);

    const char *log;
    DynamixelWorkbench driver;
    driver.init(port.c_str(), baud_rate, &log);

    follower.initialize_dynamixel();

    while (ros::ok()) {
        ros::spinOnce();

        sensor_msgs::JointState state = follower.follow_trajectory();
        joint_pub.publish(state);

        loop_rate.sleep();
    }
}