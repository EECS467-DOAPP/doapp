#include "generate_trajectories.cuh"
#include "gpu_error_check.cuh"
#include "unique_ptr.cuh"

#include <algorithm>
#include <cassert>
#include <functional>
#include <string>
#include <iterator>
#include <iostream>
#include <vector>
#include <sstream>

using namespace doapp;

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

template <typename T>
T* make_unique_array(std::size_t num_elements) {
    void* ptr;
    gpuErrchk(cudaMalloc(&ptr, num_elements * sizeof(T)));
    return static_cast<T*>(ptr);
}

struct floatCompare {
    int error_call = 0;
    bool operator()(float lhs, float rhs) {
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
        error_call += result; //only increments when lhs and rhs are considered equal
        return result;
    }
};

bool float_vectors_equal(const std::vector<float>& lhs, const std::vector<float>& rhs, const std::string& error_msg = "") {
    floatCompare comparator;
    if(!std::equal(lhs.cbegin(), lhs.cend(), rhs.cbegin(), std::ref(comparator))) {
        if(!error_msg.empty())
            std::cerr << error_msg << std::endl;
        std::cerr << "Error comparing " << lhs.at(comparator.error_call) << " to " << rhs.at(comparator.error_call) << std::endl;
        return false;
    }
    return true;
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
    std::cout << "planning trajectory from (";
    std::copy(initial_point.begin(), initial_point.end(), std::ostream_iterator<float>(std::cout, ", "));
    std::cout << ") to (";
    std::copy(final_point.begin(), final_point.end(), std::ostream_iterator<float>(std::cout, ", "));
    std::cout << ')' << std::endl;

    unsigned int num_rngs = k*d*std::max(n,m);

    gpuErrchk(cudaMemcpyToSymbol(initial_waypoint, initial_point.data(), initial_point.size() * sizeof(float), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(final_waypoint, final_point.data(), final_point.size() * sizeof(float), 0, cudaMemcpyHostToDevice));
    UniquePtr<float> dev_trajectories(make_unique_array<float>(k*n*d));
    UniquePtr<float> dev_noise_vectors(make_unique_array<float>(k*m*d));
    UniquePtr<float> dev_noisy_trajectories(make_unique_array<float>(k*m*n*d));
    UniquePtr<curandState> dev_rngs(make_unique_array<curandState>(num_rngs));
    UniquePtr<float> dev_velocities(make_unique_array<float>(k*m*n*d));
    UniquePtr<float> dev_accelerations(make_unique_array<float>(k*m*n*d));
    UniquePtr<float> dev_smoothness(make_unique_array<float>(k*m));
    UniquePtr<float> dev_scores(make_unique_array<float>(k*m));

    std::vector<float> host_trajectories(k*n*d);
    std::vector<float> host_noise_vectors(k*m*d);
    std::vector<float> host_noisy_trajectories(k*m*n*d);
    std::vector<float> host_velocities(k*m*n*d);
    std::vector<float> host_accelerations(k*m*n*d);
    std::vector<float> host_smoothness(k*m);
    std::vector<float> host_scores(k*m);

    dim3 num_threads(512);
    dim3 num_blocks(ceil(double(num_rngs)/num_threads.x));
    init_cudarand<<<num_blocks, num_threads>>>(dev_rngs.get(), num_rngs);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    dim3 gridDim(k);
    dim3 blockDim(n*m*d);
    init_traject_kernel<<<gridDim, blockDim>>>(n, d, dev_rngs.get(), d*n, dev_trajectories.get());
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(host_trajectories.data(), dev_trajectories.get(), k*n*d*sizeof(float), cudaMemcpyDeviceToHost));
    gen_noise<<<gridDim, blockDim>>>(m, d, dev_noise_vectors.get(), dev_rngs.get(), d*std::max(m, n));
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(host_noise_vectors.data(), dev_noise_vectors.get(), k*m*d*sizeof(float), cudaMemcpyDeviceToHost));

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
    init_cudarand<<<num_blocks, num_threads>>>(dev_rngs.get(), num_rngs);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    optimize_trajectories<<<gridDim, blockDim>>>(dev_trajectories.get(), dev_noise_vectors.get(), dev_noisy_trajectories.get(), dev_rngs.get(), dev_velocities.get(), dev_accelerations.get(), dev_smoothness.get(), dev_scores.get(), d*std::max(n,m), n, d, m, deltaT);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    std::vector<float> kernel_trajectories(k*n*d);
    std::vector<float> kernel_noise_vectors(k*m*d);
    std::vector<float> kernel_noisy_trajectories(k*m*n*d);
    std::vector<float> kernel_velocities(k*m*n*d);
    std::vector<float> kernel_accelerations(k*m*n*d);
    std::vector<float> kernel_smoothness(k*m);
    std::vector<float> kernel_scores(k*m);
    gpuErrchk(cudaMemcpy(kernel_trajectories.data(), dev_trajectories.get(), k*n*d*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(kernel_noise_vectors.data(), dev_noise_vectors.get(), k*m*d*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(kernel_noisy_trajectories.data(), dev_noisy_trajectories.get(), kernel_noisy_trajectories.size() * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(kernel_velocities.data(), dev_velocities.get(), kernel_velocities.size() * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(kernel_accelerations.data(), dev_accelerations.get(), kernel_accelerations.size() * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(kernel_smoothness.data(), dev_smoothness.get(), kernel_smoothness.size() * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(kernel_scores.data(), dev_scores.get(), kernel_scores.size() * sizeof(float), cudaMemcpyDeviceToHost));
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
    bool passed = true;
    passed = float_vectors_equal(kernel_trajectories, host_trajectories, "Initial trajectories do not match!") && passed;
    if(!std::all_of(kernel_trajectories.begin(), kernel_trajectories.end(), limitChecker{})) {
        std::cout << "inital trajectories exceed joint limits" << std::endl;
        passed = false;
    }
    passed = float_vectors_equal(kernel_noise_vectors, host_noise_vectors, "Noise vectors do not match!") && passed;
    if(!std::all_of(kernel_noise_vectors.begin(), kernel_noise_vectors.end(), limitChecker{})) {
        std::cout << "noise vectors exceed joint limits" << std::endl;
        passed = false;
    }
    passed = float_vectors_equal(kernel_noisy_trajectories, host_noisy_trajectories, "Noisy trajectories do not match!") && passed;
    if(!std::all_of(kernel_noisy_trajectories.begin(), kernel_noisy_trajectories.end(), limitChecker{})) {
        std::cout << "noisy trajectories exceed joint limits" << std::endl;
        passed = false;
    }
    passed = float_vectors_equal(kernel_velocities, host_velocities, "Velocities do not match!") && passed;
    passed = float_vectors_equal(kernel_accelerations, host_accelerations, "Accelerations do not match!") && passed;
    passed = float_vectors_equal(kernel_smoothness, host_smoothness, "Smoothness does not match!") && passed;
    passed = float_vectors_equal(kernel_scores, host_scores, "Scores does not match!") && passed;
    if(passed) {
        std::cout << "Passed!" << std::endl;
    } else {
        std::cout << "Failed!" << std::endl;
    }
    return passed;
}
