# DOAPP &mdash; Dynamic Object Avoidance and Path Planning

## Note: Setting up Dynamixel Workbench on ROS Melodic Morenia
The [official Dynamixel Workbench Wiki](http://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_workbench/) does not tell us exactly how to install Dynamixel Workbench on ROS. However, for Ubuntu 18.04 LTS and ROS Melodic Morenia, Dynamixel Workbench can be installed with the following command:
```bash
sudo apt-get install ros-melodic-dynamixel-workbench
```

After installation, put in the following command:
```bash
rosrun dynamixel_workbench_controllers find_dynamixel /dev/ttyUSB0
```
if your installation is successful, the following information should appear in the terminal.
```bash
[ INFO] [1574126585.648482505]: Succeed to init(9600)
[ INFO] [1574126585.648632390]: Wait for scanning...
[ INFO] [1574126607.431973937]: Find 0 Dynamixels
[ INFO] [1574126607.434631844]: Succeed to init(57600)
[ INFO] [1574126607.434702659]: Wait for scanning...
[ INFO] [1574126616.325738247]: Find 1 Dynamixels
[ INFO] [1574126616.325762620]: id : 2, model name : MX-28
[ INFO] [1574126616.327797308]: Succeed to init(115200)
[ INFO] [1574126616.327803363]: Wait for scanning...
[ INFO] [1574126633.989176665]: Find 0 Dynamixels
[ INFO] [1574126633.992513468]: Succeed to init(1000000)
[ INFO] [1574126633.992584478]: Wait for scanning...
[ INFO] [1574126642.631704338]: Find 5 Dynamixels
[ INFO] [1574126642.631722487]: id : 1, model name : MX-28
[ INFO] [1574126642.631726597]: id : 3, model name : MX-28
[ INFO] [1574126642.631729949]: id : 4, model name : AX-12A
[ INFO] [1574126642.631733386]: id : 5, model name : AX-12A
[ INFO] [1574126642.631736644]: id : 6, model name : AX-12A
[ INFO] [1574126642.634441154]: Succeed to init(2000000)
[ INFO] [1574126642.634447559]: Wait for scanning...
[ INFO] [1574126659.943597444]: Find 0 Dynamixels
[ INFO] [1574126659.947011607]: Succeed to init(3000000)
[ INFO] [1574126659.947083074]: Wait for scanning...
[ INFO] [1574126677.246625317]: Find 0 Dynamixels
[ INFO] [1574126677.250220742]: Succeed to init(4000000)
[ INFO] [1574126677.250293536]: Wait for scanning...
[ INFO] [1574126694.546787918]: Find 0 Dynamixels
```
Make sure that the USB connection between your PC and the Dynamixel motors has been established.

## Compiling Dynamixel Controller
To compile Dynamixel controller, type `catkin_make` in your catkin workspace. 

## Running Dynamixel Controller
To run Dynamixel controller, run the following command in a new terminal:
```bash
roslaunch dynamixel_driver controller.launch
```
if the node is launched successfully, the motors should all be initialized to the middle point, with the following information appearing on the last few lines of terminal:
```bash
process[dynamixel_workbench-1]: started with pid [13281]
[ INFO] [1575338505.590665958]: Name : base, ID : 0, Model Number : 29
[ INFO] [1575338505.594579404]: Name : elbow, ID : 2, Model Number : 29
[ INFO] [1575338505.596599995]: Name : gripper, ID : 5, Model Number : 12
[ INFO] [1575338505.598533763]: Name : shoulder, ID : 1, Model Number : 29
[ INFO] [1575338505.600563355]: Name : wrist, ID : 3, Model Number : 12
[ INFO] [1575338505.602562169]: Name : wrist_rotate, ID : 4, Model Number : 12
```

Then, open a new terminal and run
```bash
rosrun dynamixel_driver controller
```

## Running Dynamixel Controller Demo
A small demo of how to send commands to the controller in another ROS node is also provided in the source code. To run the demo, please [run Dynamixel controller first](#running-dynamixel-controller). Then, in a new terminal type:
```bash
rosrun dynamixel_driver controller_demo
```
You will see the 4th motor (counting from the base) starts to move back and forth.