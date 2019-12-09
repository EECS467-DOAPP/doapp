#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/String.h>

#include <dynamixel_driver/driver.h>

int main(int argc, char **argv) {
    ros::init(argc, argv, "dynamixel_controller");
    DynamixelDriver driver;
    ros::Rate loop_rate(100);  // 100Hz loop rate

    // PART 1 -- Sending initialization motor commands
    driver.send();

    // PART 2 -- Listening to motor states
    ros::Subscriber subscribeMotorState =
        driver.n_.subscribe("/dynamixel_workbench/dynamixel_state", 1000, &DynamixelDriver::handleState, &driver);

    // PART 3 -- Listening to arm command from other nodes
    ros::ServiceServer serviceArmCommand =
        driver.n_.advertiseService("arm_command", &DynamixelDriver::handleArmCommand, &driver);

    // PART 4 -- Listening to gripper command from other nodes
    ros::ServiceServer serviceGripperCommand =
        driver.n_.advertiseService("gripper_command", &DynamixelDriver::handleGripperCommand, &driver);

    // PART 5 -- Publish motor states to joint_states to RViz
    ros::Publisher pub = driver.n_.advertise<sensor_msgs::JointState>("joint_states", 1000);
    pub.publish(driver.visualization_msg);

    loop_rate.sleep();
    ros::spin();
    return 0;
}
