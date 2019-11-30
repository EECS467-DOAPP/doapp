#include <dynamixel_driver/driver.h>
#include <ros/ros.h>

/**
 * @brief   This demo shows how to send commands to the driver 
 *          from other node. The message type is MotorCommand
 *          with a goal position and an motor ID. The driver will
 *          set the goal position of the motor with corresponding 
 *          ID to be the value specified in the message.
 * 
 * @param argc: None
 * @param argv: None
 * @return int: None
 */

int main(int argc, char **argv) {
    ros::init(argc, argv, "dynamixel_driver_demo");
    ros::NodeHandle n;
    ros::Publisher pub = n.advertise<dynamixel_driver::MotorCommand>("motor_command", 1000);
    ros::Rate loop_rate(1);

    const int32_t lower_bound = 1024;
    const int32_t upper_bound = 3072;
    bool increasing = true;

    dynamixel_driver::MotorCommand msg;
    msg.id = 2;
    msg.goal = lower_bound;
    while (ros::ok()) {
        if (increasing) {
            if (msg.goal < upper_bound)
                msg.goal += 100;
            else {
                msg.goal = upper_bound;
                increasing = false;
            }
        } else {
            if (msg.goal > lower_bound)
                msg.goal -= 100;
            else {
                msg.goal = lower_bound;
                increasing = true;
            }
        }

        ROS_INFO("Sending goal %d to motor [%d]: ", msg.goal, msg.id);

        pub.publish(msg);
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
