#include "generate_trajectories.cuh"
#include "gpu_error_check.cuh"

#include <algorithm>
#include <cassert>
#include <functional>
#include <string>
#include <iostream>
#include <vector>
#include <sstream>

//host helper function to take the derivative of an input sequence
void take_derivative(float inital[], float final[], float trajectory[], unsigned int waypoint_dim, unsigned int num_waypoints, float deltaT, float output[]) {
    //initalize first waypoint
    for(unsigned int i = 0; i < waypoint_dim; ++i) {
        output[i] = (trajectory[i] - inital[i]) / deltaT;
    }
    for(unsigned int waypoint = 1; waypoint < num_waypoints; ++waypoint) {
        for(unsigned int dim = 0; dim < waypoint_dim; ++dim) {
            float current_val = trajectory[(waypoint*waypoint_dim + dim)];
            float prior_val = waypoint ? trajectory[(waypoint-1)*waypoint_dim + dim] : inital[dim];
            output[waypoint*waypoint_dim + dim] = (current_val - prior_val) / deltaT;
        }
    }
}

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
    std::vector<float> initial_point{1024.0f, 1024.0f, 1024.0f, 0.0f, 0.0f};
    std::vector<float> final_point{2000.0f, 2000.0f, 2000.0f, 892.0f, 0.0f};
    float deltaT = 1; //1 second per waypoint
    unsigned int k = 10, n = 25, m = 6, d = 5; //TODO: have m be calculated from 1024/(n*d)
    bool verbose = false;
    if(argc > 1) {
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
            } else if(argv[i] == std::string("-v")) {
                verbose = true;
            } else {
                std::cerr << "unrecognized argument: " << argv[i] << std::endl;
                return 1;
            }
        }
    }
    std::cout << "Running test with k = " << k << ", n = " << n << ", m = " << m << ", verbose: " << verbose << std::endl;
    unsigned int num_rngs = k*d*std::max(n,m);
    float *dev_trajectories, *dev_noise_vectors, *dev_noisy_trajectories, *dev_velocities, *dev_accelerations, *dev_smoothness, *dev_scores;
    curandState* dev_rngs;
    gpuErrchk(cudaMemcpyToSymbol(initial_waypoint, initial_point.data(), initial_point.size() * sizeof(float), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(final_waypoint, final_point.data(), final_point.size() * sizeof(float), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&dev_trajectories, k*n*d*sizeof(float)));
    gpuErrchk(cudaMalloc(&dev_noise_vectors, k*m*d*sizeof(float)));
    gpuErrchk(cudaMalloc(&dev_noisy_trajectories, k*m*n*d*sizeof(float)));
    gpuErrchk(cudaMalloc(&dev_rngs, num_rngs * sizeof(curandState)));
    gpuErrchk(cudaMalloc(&dev_velocities, k*m*n*d*sizeof(float)));
    gpuErrchk(cudaMalloc(&dev_accelerations, k*m*n*d*sizeof(float)));
    gpuErrchk(cudaMalloc(&dev_smoothness, k*m*sizeof(float)));
    gpuErrchk(cudaMalloc(&dev_scores, k*m*sizeof(float)));


    std::vector<float> host_trajectories(k*n*d);
    std::vector<float> host_noise_vectors(k*m*d);
    std::vector<float> host_noisy_trajectories(k*m*n*d);
    std::vector<float> host_velocities(k*m*n*d);
    std::vector<float> host_accelerations(k*m*n*d);
    std::vector<float> host_smoothness(k*m);
    std::vector<float> host_scores(k*m);

    dim3 num_threads(512);
    dim3 num_blocks(ceil(double(num_rngs)/num_threads.x));
    init_cudarand<<<num_blocks, num_threads>>>(dev_rngs, num_rngs);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    dim3 gridDim(k);
    dim3 blockDim(n*m*d);
    init_traject_kernel<<<gridDim, blockDim>>>(n, d, dev_rngs, d*n, dev_trajectories);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(host_trajectories.data(), dev_trajectories, k*n*d*sizeof(float), cudaMemcpyDeviceToHost));
    gen_noise<<<gridDim, blockDim>>>(m, d, dev_noise_vectors, dev_rngs, d*std::max(m, n));
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(host_noise_vectors.data(), dev_noise_vectors, k*m*d*sizeof(float), cudaMemcpyDeviceToHost));

    //given inital trajectories and noise vectors, compute noisy trajectories
    for(unsigned int traj = 0; traj < k; ++traj) {
        for(unsigned int noise = 0; noise < m; ++noise) {
            for(unsigned int waypoint = 0; waypoint < n; ++waypoint) {
                for(unsigned int dim = 0; dim < d; ++dim) {
                    host_noisy_trajectories[traj * (m*n*d) + noise*(n*d) + waypoint*d + dim] = host_trajectories[traj*(n*d) + waypoint*d + dim] + host_noise_vectors[traj*m*d + noise*d + dim];
                }
            }
        }
    }
    //compute joint velocities for each noisy trajectory
    for(unsigned int i = 0; i < k*m; ++i) {
        take_derivative(initial_point.data(), final_point.data(), host_noisy_trajectories.data() + i*n*d, d, n, deltaT, host_velocities.data() + i*n*d);
    }
    //compute joint accelerations for each noisy trajectory
    for(unsigned int i = 0; i < k*m; ++i) {
        take_derivative(initial_point.data(), final_point.data(), host_velocities.data() + i*n*d, d, n, deltaT, host_accelerations.data() + i*n*d);
    }
    //compute smoothness score for each trajectory
    for(unsigned int traj = 0; traj < k*m; ++traj) {
        host_smoothness[traj] = 0;
        for(unsigned int waypoint = 0; waypoint < n; ++waypoint) {
            for(unsigned int dim = 0; dim < d; ++dim) {
                host_smoothness[traj] += host_accelerations[traj*n*d + waypoint*d + dim] * host_accelerations[traj*n*d + waypoint*d + dim];
            }
        }
        host_smoothness[traj] *= 0.5;
    }
    //score each trajectory
    //TODO: host collision detection?
    std::copy(host_smoothness.begin(), host_smoothness.end(), host_scores.begin());

    //reset the cudarand state should reset the rngs to the start of whatever sequence they were on
    init_cudarand<<<num_blocks, num_threads>>>(dev_rngs, num_rngs);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    optimize_trajectories<<<gridDim, blockDim>>>(dev_trajectories, dev_noise_vectors, dev_noisy_trajectories, dev_rngs, dev_velocities, dev_accelerations, dev_smoothness, dev_scores, d*std::max(n,m), n, d, m, deltaT);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    std::vector<float> kernel_trajectories(k*n*d);
    std::vector<float> kernel_noise_vectors(k*m*d);
    std::vector<float> kernel_noisy_trajectories(k*m*n*d);
    std::vector<float> kernel_velocities(k*m*n*d);
    std::vector<float> kernel_accelerations(k*m*n*d);
    std::vector<float> kernel_smoothness(k*m);
    std::vector<float> kernel_scores(k*m);
    gpuErrchk(cudaMemcpy(kernel_trajectories.data(), dev_trajectories, k*n*d*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(kernel_noise_vectors.data(), dev_noise_vectors, k*m*d*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(kernel_noisy_trajectories.data(), dev_noisy_trajectories, kernel_noisy_trajectories.size() * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(kernel_velocities.data(), dev_velocities, kernel_velocities.size() * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(kernel_accelerations.data(), dev_accelerations, kernel_accelerations.size() * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(kernel_smoothness.data(), dev_smoothness, kernel_smoothness.size() * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(kernel_scores.data(), dev_scores, kernel_scores.size() * sizeof(float), cudaMemcpyDeviceToHost));
    if(verbose) {
        std::cout << "Host Initial Trajectories:" << std::endl;
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
        std::cout << "Host Noise vectors:" << std::endl;
        for(unsigned int i = 0; i < k; ++i) {
            for(unsigned int waypoint = 0; waypoint < m; ++waypoint) {
                std::cout << '[';
                for(unsigned int j = 0; j < d-1; ++j) {
                    std::cout << host_noise_vectors[i*m*d + j] << ", ";
                }
                std::cout << host_noise_vectors[(i+1)*m*d - 1] << "] ";
            }
            std::cout << std::endl;
        }

        std::cout << "Host noisy trajectories:" << std::endl;
        for(unsigned int i = 0; i < k*m; ++i) {
            for(unsigned int waypoint = 0; waypoint < n; ++waypoint) {
                std::cout << '(';
                for(unsigned int j = 0; j < d-1; ++j) {
                    std::cout << host_noisy_trajectories[i*n*d + j] << ", ";
                }
                std::cout << host_noisy_trajectories[(i+1)*n*d - 1] << ") ";
            }
            std::cout << std::endl;
        }
        std::cout << "Host velocities:" << std::endl;
        for(unsigned int i = 0; i < k*m; ++i) {
            for(unsigned int waypoint = 0; waypoint < n; ++waypoint) {
                std::cout << '(';
                for(unsigned int j = 0; j < d-1; ++j) {
                    std::cout << host_velocities[i*n*d + waypoint*d + j] << ", ";
                }
                std::cout << host_velocities[(i+1)*n*d - 1] << ") ";
            }
            std::cout << std::endl;
        }
        std::cout << "Host accelerations:" << std::endl;
        for(unsigned int i = 0; i < k*m; ++i) {
            for(unsigned int waypoint = 0; waypoint < n; ++waypoint) {
                std::cout << '(';
                for(unsigned int j = 0; j < d-1; ++j) {
                    std::cout << host_accelerations[i*n*d + waypoint*d + j] << ", ";
                }
                std::cout << host_accelerations[(i+1)*n*d - 1] << ") ";
            }
            std::cout << std::endl;
        }
        std::cout << "Host smoothness:" << std::endl;
        for(unsigned int i = 0; i < k; ++i) {
            for(unsigned int j = 0; j < m; ++j) {
                std::cout << host_smoothness[j] << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << "Kernel Initial trajectories" << std::endl;
        for(unsigned int i = 0; i < k; ++i) {
            for(unsigned int waypoint = 0; waypoint < n; ++waypoint) {
                std::cout << '(';
                for(unsigned int j = 0; j < d-1; ++j) {
                    std::cout << kernel_trajectories[i*n*d + j] << ", ";
                }
                std::cout << kernel_trajectories[(i+1)*n*d - 1] << ") ";
            }
            std::cout << std::endl;
        }
        std::cout << "Kernel Noise vectors:" << std::endl;
        for(unsigned int i = 0; i < k; ++i) {
            for(unsigned int waypoint = 0; waypoint < m; ++waypoint) {
                std::cout << '[';
                for(unsigned int j = 0; j < d-1; ++j) {
                    std::cout << kernel_noise_vectors[i*m*d + j] << ", ";
                }
                std::cout << kernel_noise_vectors[(i+1)*m*d - 1] << "] ";
            }
            std::cout << std::endl;
        }

        std::cout << "Kernel noisy trajectories:" << std::endl;
        for(unsigned int i = 0; i < k*m; ++i) {
            for(unsigned int waypoint = 0; waypoint < n; ++waypoint) {
                std::cout << '(';
                for(unsigned int j = 0; j < d-1; ++j) {
                    std::cout << kernel_noisy_trajectories[i*n*d + j] << ", ";
                }
                std::cout << kernel_noisy_trajectories[(i+1)*n*d - 1] << ") ";
            }
            std::cout << std::endl;
        }
        std::cout << "Kernel velocities:" << std::endl;
        for(unsigned int i = 0; i < k*m; ++i) {
            for(unsigned int waypoint = 0; waypoint < n; ++waypoint) {
                std::cout << '(';
                for(unsigned int j = 0; j < d-1; ++j) {
                    std::cout << kernel_velocities[i*n*d + waypoint*d + j] << ", ";
                }
                std::cout << kernel_velocities[(i+1)*n*d - 1] << ") ";
            }
            std::cout << std::endl;
        }
        std::cout << "Kernel smoothness:" << std::endl;
        for(unsigned int i = 0; i < k; ++i) {
            for(unsigned int j = 0; j < m; ++j) {
                std::cout << kernel_smoothness[j] << ", ";
            }
            std::cout << std::endl;
        }
    }
    struct limitChecker {
        int index = 0;
        bool operator()(float angle) {
            bool result = angle > doapp::host_min_joint_angles[index] && angle < doapp::host_max_joint_angles[index];
            index = ++index % doapp::num_joints;
            return result;
        }
    };
    int error_call = 0;
    std::function<bool (float, float)> floatCompare = [&](float lhs, float rhs) {
        float epsilon = fabs(lhs / std::pow(2, 32));
        bool result = std::abs(rhs - lhs) < epsilon;
        if(!result) {
            //gross heuristic: if they print the same, then it's close enough
            std::stringstream ss;
            ss << lhs;
            std::string lhs_s = ss.str();
            ss.clear();
            ss << rhs;
            std::string rhs_s = ss.str();
            result = lhs_s != rhs_s;
        }
        if(!result) {
            std::cout << "Error comparing " << lhs << " to " << rhs << std::endl;
            std::cout << "error on call: " << error_call << std::endl;
        } else
            ++error_call;
        return result;
    };
    bool passed = true;
    if(!std::equal(kernel_trajectories.begin(), kernel_trajectories.end(), host_trajectories.begin(), floatCompare)) {
        std::cout << "Trajectories do not match!" << std::endl;
        passed = false;
    }
    if(!std::all_of(kernel_trajectories.begin(), kernel_trajectories.end(), limitChecker{})) {
        std::cout << "inital trajectories exceed joint limits" << std::endl;
        passed = false;
    }
    if(!std::equal(kernel_noise_vectors.begin(), kernel_noise_vectors.end(), host_noise_vectors.begin(), floatCompare)) {
        std::cout << "noise vectors do not match!" << std::endl;
        passed = false;
    }
    if(!std::all_of(kernel_noise_vectors.begin(), kernel_noise_vectors.end(), limitChecker{})) {
        std::cout << "noise vectors exceed joint limits" << std::endl;
        passed = false;
    }
    if(!std::equal(kernel_noisy_trajectories.begin(), kernel_noisy_trajectories.end(), host_noisy_trajectories.begin(), floatCompare)) {
        std::cout << "The noisy trajectories are not equal" << std::endl;
        passed = false;
    }
    if(!std::all_of(kernel_noisy_trajectories.begin(), kernel_noisy_trajectories.end(), limitChecker{})) {
        std::cout << "noisy trajectories exceed joint limits" << std::endl;
        passed = false;
    }
    if(!std::equal(kernel_velocities.begin(), kernel_velocities.end(), host_velocities.begin(), floatCompare)) {
        std::cout << "The velocities are not equal" << std::endl;
        std::cout << "error comparing " << kernel_velocities[error_call-1] << " to " << host_velocities[error_call-1] << ", the difference is: " << std::abs(host_velocities[error_call] - kernel_velocities[error_call]) << std::endl;
        std::cout << "prior value on host: " << host_noisy_trajectories[error_call-6] << ", prior value on kernel: " << kernel_noisy_trajectories[error_call-6] << ", current value on host: " << host_noisy_trajectories[error_call-1] << ", current value on kernel: " << kernel_noisy_trajectories[error_call-1] << std::endl;
        passed = false;
    }
    if(!std::equal(kernel_accelerations.begin(), kernel_accelerations.end(), host_accelerations.begin(), floatCompare)) {
        std::cout << "the accelerations are not equal!" << std::endl;
        passed = false;
    }
    if(!std::equal(kernel_smoothness.begin(), kernel_smoothness.end(), host_smoothness.begin(), floatCompare)) {
        std::cout << "the smoothness costs are not equal!" << std::endl;
        passed = false;
    }
    if(!std::equal(kernel_scores.begin(), kernel_scores.end(), host_scores.begin(), floatCompare)) {
        std::cout << "the scores are not equal!" << std::endl;
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
    cudaFree(dev_velocities);
    cudaFree(dev_accelerations);
    cudaFree(dev_smoothness);
    cudaFree(dev_scores);
    return passed;
}
