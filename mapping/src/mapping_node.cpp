#include <ros/ros.h>
#include <nodelet/nodelet.h>

#include <occupancy_grid/occupancy_grid.hpp>

namespace doapp
{

class MappingNode : public nodelet::Nodelet
{
public:
    MappingNode() : Nodelet(), map_(size_, granularity_)
    {
    }

    virtual void onInit()
    {
        NODELET_INFO("STARTING MAPPING NODELET");
    }

private:
    // TODO: make nodelet argument
    constexpr static double size_ = 1.0;
    constexpr static double granularity_ = 0.05;

    doapp::OccupancyGrid map_;

    ros::Subscriber sub_;
    ros::Publisher pub_;
    ros::Timer timer_;
};

} // namespace doapp

PLUGINLIB_EXPORT_CLASS(doapp::MappingNode, nodelet::Nodelet)

//                     int
//                     main(int argc, char *argv[])
// {
//     ros::init(argc, argv, "doapp_mapping");

//     ros::NodeHandle n;

//     // ros::Publisher occupancy_grid_pub = ;

//     return 0;
// }