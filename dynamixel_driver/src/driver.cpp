#include "dynamixel_driver/MotorCommand.h"
#include "dynamixel_workbench_msgs/DynamixelCommand.h"
#include "dynamixel_workbench_msgs/DynamixelStateList.h"
#include "ros/ros.h"
#include "std_msgs/String.h"

/**
 * Listens to the state of the motors in real time
 */
void handleState(const dynamixel_workbench_msgs::DynamixelStateList &msg) {
    size_t stateListSize = msg.dynamixel_state.size();
    ROS_INFO("State list length: %d", stateListSize);

    for (size_t i = 0; i < stateListSize; ++i) {
        dynamixel_workbench_msgs::DynamixelState state = msg.dynamixel_state[i];
        ROS_INFO("Current state [%d]: %d", state.id, state.present_position);
    }
}

/**
 * Listens to the motor command messages sent from other nodes
 */
void handleCommand(const dynamixel_driver::MotorCommand &msg) {
    ROS_INFO("Received motor command at [%d]: %d", msg.id, msg.goal);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "dynamixel_controller");
    ros::NodeHandle n;

    // PART 1 -- Sending service requests
    ros::ServiceClient client =
        n.serviceClient<dynamixel_workbench_msgs::DynamixelCommand>("/dynamixel_workbench/dynamixel_command");

    dynamixel_workbench_msgs::DynamixelCommand srv;
    srv.request.id = 1;
    srv.request.addr_name = "Goal_Position";
    srv.request.value = 2048;

    if (client.call(srv)) {
        ROS_INFO("Result: %d", srv.response.comm_result);
    } else {
        ROS_ERROR("Failed to call service dynamixel_command");
    }

    // PART 2 -- Listening to motor states
    ros::Subscriber subscribeMotorState = n.subscribe("/dynamixel_workbench/dynamixel_state", 1000, handleState);

    // PART 3 -- Listening to motor command from other states
    ros::Subscriber subcribeMotorCommand = n.subscribe("motor_command", 1000, handleCommand);

    ros::spin();
    return 0;
}
