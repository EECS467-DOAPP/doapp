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

namespace demo
{
/**
 * @brief Initialize message from an empty vector
 * 
 * @param msg:  message to be initialized
 */
void init(dynamixel_driver::MotorCommand &msg)
{
    for (int i = 0; i < NUM_MOTORS; ++i)
    {
        msg.commands.push_back((i < NUM_TYPE_SWITCH) ? 2048 : 512);
    }
}
} // namespace demo

int main(int argc, char **argv)
{
    ros::init(argc, argv, "dynamixel_driver_demo");
    ros::NodeHandle n;
    ros::Publisher pub = n.advertise<dynamixel_driver::MotorCommand>("motor_command", 1000);
    ros::Rate loop_rate(20); // 50Hz loop rate

    /* For MX-28 motor testing */
    // const int32_t lower_bound = 1024;
    // const int32_t upper_bound = 3072;

    /* For AX-12A motor testing */
    const int32_t lower_bound = 205;
    const int32_t upper_bound = 819;
    bool increasing = true;

    dynamixel_driver::MotorCommand msg;
    demo::init(msg);
    int id = 3;
    msg.commands[id] = lower_bound;
    while (ros::ok())
    {
        if (increasing)
        {
            if (msg.commands[id] < upper_bound)
                msg.commands[id] += 1;
            else
            {
                msg.commands[id] = upper_bound;
                increasing = false;
            }
        }
        else
        {
            if (msg.commands[id] > lower_bound)
                msg.commands[id] -= 1;
            else
            {
                msg.commands[id] = lower_bound;
                increasing = true;
            }
        }

        ROS_INFO("Sending goal %d to motor [%d]: ", msg.commands[id], id);

        pub.publish(msg);
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}