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