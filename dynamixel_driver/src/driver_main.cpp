#include <ros/ros.h>
#include <std_msgs/String.h>

#include <dynamixel_driver/driver.h>

int main(int argc, char **argv) {
    ros::init(argc, argv, "dynamixel_controller");
    DynamixelDriver driver;
    ros::Rate loop_rate(100);

    // PART 1 -- Sending initialization motor commands
    driver.send();

    // PART 2 -- Listening to motor states
    ros::Subscriber subscribeMotorState =
        driver.n_.subscribe("/dynamixel_workbench/dynamixel_state", 1000, &DynamixelDriver::handleState, &driver);

    // PART 3 -- Listening to motor command from other states
    ros::Subscriber subcribeMotorCommand =
        driver.n_.subscribe("motor_command", 1000, &DynamixelDriver::handleCommand, &driver);

    loop_rate.sleep();
    ros::spin();
    return 0;
}
