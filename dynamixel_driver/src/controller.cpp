#include <dynamixel_driver/driver.h>

/**
 * @brief initialize service client for sending commands
 * 
 */
void DynamixelDriver::init() {
    reachedNextGoal = false;
    client_ = n_.serviceClient<dynamixel_workbench_msgs::DynamixelCommand>("/dynamixel_workbench/dynamixel_command");
    for (int i = 0; i < NUM_MOTORS; ++i) {
        visualization_msg.position.push_back(0.0f);
    }
}

/**
 * @brief update message sent to visualization
 * 
 */
void DynamixelDriver::setMsg() {
    for (int i = 0; i < NUM_MOTORS; ++i) {
        switch (cmd[i].type) {
            case MotorType::MX_28:
                visualization_msg.position[i] = cmd[i].goalPosition / 4096.0;
                break;
            case MotorType::AX_12A:
                visualization_msg.position[i] = cmd[i].goalPosition / 1024.0;

            default:
                break;
        }
    }
}

/**
 * @brief   Default constructor.
 *          Constructs driver with all motors initialized to middle points.
 * 
 */
DynamixelDriver::DynamixelDriver() {
    init();
    // push back goal position commands
    for (int i = 0; i < NUM_MOTORS; ++i) {
        DynamixelDriverGoal goal;
        if (i < NUM_TYPE_SWITCH) {
            goal.type = MotorType::MX_28;
            goal.goalPosition = 2048;
        } else {
            goal.type = MotorType::AX_12A;
            goal.goalPosition = 512;
        }

        cmd.push_back(goal);
    }

    setMsg();
    // construct client and subscriber
}

/**
     * @brief Construct a new Dynamixel Driver object
     * 
     * @param goalList: complete list of goal positions for six Dynamixel motors
     */
DynamixelDriver::DynamixelDriver(std::vector<DynamixelDriverGoal>& goalList) {
    init();
    cmd = goalList;
    setMsg();
}

/**
     * @brief Construct a new Dynamixel Driver object
     * 
     * @param id:       id of Dynamixel motor
     * @param goalpos:  goal position for Dynamixel motor #[id]
     */
DynamixelDriver::DynamixelDriver(const uint8_t id, const int32_t goalpos) {
    init();
    for (uint8_t i = 0; i < NUM_MOTORS; ++i) {
        DynamixelDriverGoal goal;

        if (i < NUM_TYPE_SWITCH)
            goal.type = MotorType::MX_28;
        else
            goal.type = MotorType::AX_12A;

        if (i == id)
            goal.goalPosition = goalpos;
        else
            goal.goalPosition = (goal.type == MotorType::MX_28) ? 2048 : 512;

        cmd.push_back(goal);
    }

    setMsg();
}

/**
 * @brief send the list of commands to Rexarm
 * 
 */
void DynamixelDriver::send() {
    dynamixel_workbench_msgs::DynamixelCommand srv;
    for (int i = 0; i < NUM_MOTORS; ++i) {
        srv.request.id = i;
        srv.request.addr_name = "Goal_Position";
        srv.request.value = cmd[i].goalPosition;

        if (client_.call(srv)) {
            ROS_DEBUG("Result: %d", srv.response.comm_result);
        } else {
            ROS_ERROR("Failed to call service dynamixel_command");
        }
    }
}

/**
 * @brief Sets the next goal of a specific motor
 * 
 * @param id:        motor ID
 * @param goalpos:   motor goal position
 */
void DynamixelDriver::set(const uint8_t id, const int32_t goalpos) {
    cmd[id].goalPosition = goalpos;
}

/**
 * @brief Sets the next goal of a specific motor
 * 
 * @param goals:    list of goals received from ROS message
 */
void DynamixelDriver::setArm(const std::vector<int>& goals) {
    cmd.clear();
    for (int i = 0; i < NUM_ARM_MOTORS; ++i) {
        DynamixelDriverGoal goal;
        goal.type = (i < NUM_TYPE_SWITCH) ? MotorType::MX_28 : MotorType::AX_12A;
        goal.goalPosition = goals[i];
        cmd.push_back(goal);
    }
}

void DynamixelDriver::setGripper(const int32_t goalpos) {
    cmd.back().goalPosition = goalpos;
}

/**
 * @brief   Receives and handles state messages from motor.
 *          Currently just prints out the state.
 * 
 * @param msg:  received ROS messages
 */
void DynamixelDriver::handleState(const dynamixel_workbench_msgs::DynamixelStateList& msg) {
    size_t stateListSize = msg.dynamixel_state.size();
    ROS_INFO("State list length: %d", stateListSize);

    reachedNextGoal = true;
    int threshold = 15;
    for (size_t i = 0; i < stateListSize; ++i) {
        dynamixel_workbench_msgs::DynamixelState state = msg.dynamixel_state[i];
        ROS_INFO("Current state [%d]: %d", state.id, state.present_position);

        if (abs(state.present_position - cmd[state.id].goalPosition) > threshold)
            // If any of the current motor state differs from the goal above a threshold,
            //  it has not reached the target.
            reachedNextGoal = false;
    }
}

/**
 * @brief   Receives and handles arm command messages sent from other nodes.
 *          Sets the new goal position for each motors
 * 
 * @param req:  goal positions sent from other nodes 
 * @param res:  indicator of reaching the goal position
 */
bool DynamixelDriver::handleArmCommand(dynamixel_driver::ArmCommand::Request& req,
                                       dynamixel_driver::ArmCommand::Response& res) {
    for (int i = 0; i < NUM_ARM_MOTORS; ++i)
        ROS_DEBUG("Received arm command %d at [%d]\n", req.commands[i], i);
    setArm(req.commands);
    send();
    setMsg();
    res.reached = reachedNextGoal;

    return true;
}

/**
 * @brief       Receives and handles gripper command messages sent from other nodes.
 *              Sets the new goal position for gripper motor only.
 * 
 * @param req:  goal position sent from other nodes
 * @param res:  indicator of reaching the goal 
 * @return true:    handling command successful 
 * @return false:   handling command unsuccessful (currently DEPRECATED)
 */
bool DynamixelDriver::handleGripperCommand(dynamixel_driver::GripperCommand::Request& req,
                                           dynamixel_driver::GripperCommand::Response& res) {
    ROS_DEBUG("Received gripper command %d\n", req.command);
    setGripper(req.command);
    send();
    setMsg();
    res.reached = reachedNextGoal;

    return true;
}