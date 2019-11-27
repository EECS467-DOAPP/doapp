#ifndef __DRIVER_H
#define __DRIVER_H

#include <vector>
#include "ros/ros.h"

/**
 * @brief Wrapper of goal position for a specific Dynamixel motor
 * 
 */
struct DynamixelDriverGoal {
    uint8_t id;
    int32_t goalPosition;
};

/**
 * @brief Vector of goal positions of the six Dynamixel motors
 * 
 */
class DynamixelDriver {
private:
    std::vector<DynamixelDriverGoal> cmd;

public:
    DynamixelDriver();
    /**
     * @brief Construct a new Dynamixel Driver object
     * 
     * @param goalList: complete list of goal positions for six Dynamixel motors
     */
    DynamixelDriver(std::vector<DynamixelDriverGoal>& goalList);
    /**
     * @brief Construct a new Dynamixel Driver object
     * 
     * @param id:    id of Dynamixel motor
     * @param goal:  goal position for Dynamixel motor #[id]
     */
    DynamixelDriver(const uint8_t id, const int32_t goal);
    ~DynamixelDriver();
};

DynamixelDriver::DynamixelDriver() {
}

DynamixelDriver::~DynamixelDriver() {
}

#endif