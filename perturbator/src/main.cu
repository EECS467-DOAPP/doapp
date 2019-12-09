#include "gpu_error_check.cuh"
#include "generate_trajectories.cuh"
#include "common.cuh"
#include "vector.cuh"

#include "ros/ros.h"
#include "sensor_msgs/JointState.h"


constexpr float deltaT = 1; //seconds. Can be tuned
struct PlanningRequest {
    bool has_goal = false;
    std::vector<float> goal_state{doapp::num_joints};
    std::vector<float> current_state{doapp::num_joints};
} planning_requestor;

int main(int argc, char** argv) {
    ros::init(argc, argv, "motion_planner");

    ros::NodeHandle node_handle;

    ros::Publisher publisher = node_handle.advertise<std_msgs::String>("motor_command", 1000);

    //no clue what loop_rate should be
    ros::Rate loop_rate(10);

    unsigned int k = 10, n = 25, m = 6, d = 5; //TODO: have m be calculated from 1024/(n*d)
    //also TODO: have k, n, m, d be grabbed from ParamServer
    std::cout << "Running with k = " << k << ", n = " << n << ", m = " << m << std::endl;
    unsigned int num_rngs = k*d*std::max(n,m);

    doapp::Vector<float> trajectories(k*n*d);
    doapp::Vector<float> noise_vectors(k*m*d);
    doapp::Vector<float> noisy_trajectories(k*m*n*d);
    doapp::Vector<curandState> rngs(num_rngs);
    doapp::Vector<float> velocities(k*m*n*d);
    doapp::Vector<float> accelerations(k*m*n*d);
    doapp::Vector<float> smoothness(k*m*n*d);
    doapp::Vector<float> scores(k*m*n*d);
    dim3 gridDim(k);
    dim3 blockDim(n*m*d);
    //initalize rngs
    init_cudarand<<<ceil(double(num_rngs)/double(512)), 512>>>(rngs.get(), num_rngs);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    while(ros::ok()) {
        if(planning_requestor.has_goal) {
            gpuErrchk(cudaMemcpyToSymbol(initial_waypoint, planning_requestor.current_state.data(), planning_requestor.current_state.size() * sizeof(float), 0, cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpyToSymbol(final_waypoint, planning_requestor.goal_state.data(), planning_requestor.goal_state.size() * sizeof(float), 0, cudaMemcpyHostToDevice));
            optimize_trajectories<<<gridDim, blockDim>>>(trajectories.get(), noise_vectors.get(), noisy_trajectories.get(), rngs.get(), velocities.get(), accelerations.get(), smoothness.get(), scores.get(), update_vector.get(), d*std::max(n,m), n, d, m, deltaT);
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());
        }
    }
}
