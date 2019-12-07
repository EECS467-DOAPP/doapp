#include "generate_trajectories.cuh"
#include "gpu_error_check.cuh"

#include <algorithm>
#include <functional>
#include <string>
#include <iostream>
#include <vector>

//wrapper around device function initalize_trajectories to be unit testable
__global__ void init_traject_kernel(unsigned int num_waypoints, unsigned int waypoint_dim, curandState* rngs, unsigned int num_rngs_per_block, float* all_trajectories) {
    float* trajectories_for_my_block = all_trajectories + (blockIdx.x * num_waypoints * waypoint_dim);
    curandState* my_rng = threadIdx.x < num_rngs_per_block ? rngs+(blockIdx.x*num_rngs_per_block+threadIdx.x) : nullptr;
    initalize_trajectories(num_waypoints, waypoint_dim, my_rng, trajectories_for_my_block);
}

//wrapper around device function to generate noise vectors
__global__ void gen_noise(unsigned int num_noise_vectors, unsigned int noise_vector_dim, float* all_noise_vectors, curandState* states, unsigned int num_rngs_per_trajectory) {
    float* my_noise_vectors = all_noise_vectors + (blockIdx.x * num_noise_vectors * noise_vector_dim);
    curandState* rng = threadIdx.x < num_rngs_per_trajectory ?  states + (blockIdx.x * num_rngs_per_trajectory + threadIdx.x) : nullptr;
    generate_noise_vectors(num_noise_vectors, noise_vector_dim, my_noise_vectors, rng);
}

int main(int argc, char** argv) {
    unsigned int k = 10, n = 25, m = 6, d = 5; //TODO: have m be calculated from 1024/(n*d)
    if(argc > 2) {
        if(argv[1] == std::string("-h") || argv[1] == std::string("--help")) {
            std::cout << "Usage: " << argv[0] << " [-n <num_waypoints> | -k <num_trajectories> | -m <num_noise_vectors>] [-n <num_waypoints> | -k <num_trajectories> | -m <num_noise_vectors>] [-n <num_waypoints> | -k <num_trajectories> | -m <num_noise_vectors>]" << std::endl;
            return 0;
        }
        for(int i = 1; i < argc; ++i) {
            if(argv[i] == std::string("-n")) {
                if(!argv[i+1]) {
                    std::cerr << "error! must specify an arg with -n" << std::endl;
                    return 1;
                }
                n = std::stoi(argv[++i]);
            } else if(argv[i] == std::string("-k")) {
                if(!argv[i+1]) {
                    std::cerr << "error! must specify an arg with -k" << std::endl;
                    return 1;
                }
                k = std::stoi(argv[++i]);
            } else if(argv[i] == std::string("-m")) {
                if(!argv[i+1]) {
                    std::cerr << "error! must specify an arg with -m" << std::endl;
                    return 1;
                }
                m = std::stoi(argv[++i]);
            } else {
                std::cerr << "unrecognized argument: " << argv[i] << std::endl;
                return 1;
            }
        }
    }
    std::cout << "Running test with k = " << k << ", n = " << n << ", m = " << m << std::endl;
    unsigned int num_rngs = k*d*std::max(n,m);
    float *dev_trajectories, *dev_noise_vectors, *dev_noisy_trajectories;
    curandState* dev_rngs;
    gpuErrchk(cudaMalloc(&dev_trajectories, k*n*d*sizeof(float)));
    gpuErrchk(cudaMalloc(&dev_noise_vectors, k*m*d*sizeof(float)));
    gpuErrchk(cudaMalloc(&dev_noisy_trajectories, k*m*n*d*sizeof(float)));
    gpuErrchk(cudaMalloc(&dev_rngs, num_rngs * sizeof(curandState)));


    std::vector<float> host_trajectories(k*n*d);
    std::vector<float> host_noise_vectors(k*m*d);
    std::vector<float> host_noisy_trajectories(k*m*n*d);

    dim3 num_blocks(ceil(double(num_rngs)/1024.0));
    dim3 num_threads(1024);
    init_cudarand<<<num_blocks, num_threads>>>(dev_rngs, num_rngs);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    dim3 gridDim(k);
    dim3 blockDim(n*m*d);
    init_traject_kernel<<<gridDim, blockDim>>>(n, d, dev_rngs, d*n, dev_trajectories);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(host_trajectories.data(), dev_trajectories, k*n*d*sizeof(float), cudaMemcpyDeviceToHost));
    gen_noise<<<gridDim, blockDim>>>(m, d, dev_noise_vectors, dev_rngs, d*m);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(host_noise_vectors.data(), dev_noise_vectors, k*m*d*sizeof(float), cudaMemcpyDeviceToHost));

    for(unsigned int traj = 0; traj < k; ++traj) {
        for(unsigned int noise = 0; noise < m; ++noise) {
            for(unsigned int waypoint = 0; waypoint < n; ++waypoint) {
                for(unsigned int dim = 0; dim < d; ++dim) {
                    host_noisy_trajectories[traj * (m*n*d) + noise*(n*d) + waypoint*d + d] = host_trajectories[traj*(n*d) + waypoint*d + dim] + host_noise_vectors[traj*m*d + noise*d + dim];
                }
            }
        }
    }

    std::cout << "Inital Trajectories:" << std::endl;
    for(unsigned int i = 0; i < k; ++i) {
        for(unsigned int waypoint = 0; waypoint < n; ++waypoint) {
            std::cout << '(';
            for(unsigned int j = 0; j < d-1; ++j) {
                std::cout << host_trajectories[i*n*d + j] << ", ";
            }
            std::cout << host_trajectories[(i+1)*n*d - 1] << ") ";
        }
        std::cout << std::endl;
    }

    std::cout << "Noise vectors:" << std::endl;
    for(unsigned int i = 0; i < k; ++i) {
        for(unsigned int waypoint = 0; waypoint < m; ++waypoint) {
            std::cout << '[';
            for(unsigned int j = 0; j < d-1; ++j) {
                std::cout << host_trajectories[i*n*d + j] << ", ";
            }
            std::cout << host_trajectories[(i+1)*n*d - 1] << "] ";
        }
        std::cout << std::endl;
    }
    //reset the cudarand state should reset the rngs to the start of whatever sequence they were on
    init_cudarand<<<num_blocks, num_threads>>>(dev_rngs, num_rngs);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    optimize_trajectories<<<gridDim, blockDim>>>(dev_trajectories, dev_noise_vectors, dev_noisy_trajectories, dev_rngs, d*std::max(n,m), n, d, m);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    std::vector<float> kernel_trajectories(k*n*d);
    std::vector<float> kernel_noise_vectors(k*m*d);
    std::vector<float> kernel_noisy_trajectories(k*m*n*d);
    gpuErrchk(cudaMemcpy(kernel_trajectories.data(), dev_trajectories, k*n*d*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(kernel_noise_vectors.data(), dev_noise_vectors, k*m*d*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(kernel_noisy_trajectories.data(), dev_noisy_trajectories, kernel_noisy_trajectories.size() * sizeof(float), cudaMemcpyDeviceToHost));
    struct limitChecker {
        int index = 0;
        bool operator()(float angle) {
            bool result = angle > doapp::host_min_joint_angles[index] && angle < doapp::host_max_joint_angles[index];
            index = ++index % doapp::num_joints;
            return result;
        }
    };
    std::function<bool (float, float)> floatCompare = [](float lhs, float rhs) {
        return std::abs(rhs - lhs) < 0.001f;
    };
    bool passed = true;
    if(std::equal(kernel_trajectories.begin(), kernel_trajectories.end(), host_trajectories.begin(), floatCompare)) {
        std::cout << "Trajectories do not match!" << std::endl;
        passed = false;
    }
    if(!std::all_of(kernel_trajectories.begin(), kernel_trajectories.end(), limitChecker{})) {
        std::cout << "inital trajectories exceed joint limits" << std::endl;
        passed = false;
    }
    if(std::equal(kernel_noise_vectors.begin(), kernel_noise_vectors.end(), host_noise_vectors.begin(), floatCompare)) {
        std::cout << "noise vectors do not match!" << std::endl;
        passed = false;
    }
    if(!std::all_of(kernel_noise_vectors.begin(), kernel_noise_vectors.end(), limitChecker{})) {
        std::cout << "noise vectors exceed joint limits" << std::endl;
        passed = false;
    }
    if(std::equal(kernel_noisy_trajectories.begin(), kernel_noisy_trajectories.end(), host_noisy_trajectories.begin(), floatCompare)) {
        std::cout << "They are not equal!" << std::endl;
        passed = false;
    }
    if(!std::all_of(kernel_noisy_trajectories.begin(), kernel_noisy_trajectories.end(), limitChecker{})) {
        std::cout << "noisy trajectories exceed joint limits" << std::endl;
        passed = false;
    }
    if(passed) {
        std::cout << "Passed!" << std::endl;
    } else {
        std::cout << "Failed!" << std::endl;
    }
    cudaFree(dev_trajectories);
    cudaFree(dev_noise_vectors);
    cudaFree(dev_noisy_trajectories);
    cudaFree(dev_rngs);
    return passed;
}
