#include "generate_trajectories.cuh"
#include "gpu_error_check.cuh"

#include <string>
#include <iostream>

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
    unsigned int k = 10, n = 25, m = 6, d = 6;
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

    float *host_trajectories = new float[k*n*d];
    float *host_noise_vectors = new float[k*m*d];
    float *host_noisy_trajectories = new float[k*m*n*d];

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
    gpuErrchk(cudaMemcpy(host_trajectories, dev_trajectories, k*n*d*sizeof(float), cudaMemcpyDeviceToHost));
    gen_noise<<<gridDim, blockDim>>>(m, d, dev_noise_vectors, dev_rngs, d*m);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(host_noise_vectors, dev_noise_vectors, k*m*d*sizeof(float), cudaMemcpyDeviceToHost));
}
