#include "common.cuh"

#include <limits>
#include <iostream>
#include <cfloat>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    std::cerr << "Error at " << __FILE__ << ':' << __LINE__ << std::endl;\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    std::cerr << "Error at " << __FILE__ << ':' << __LINE__ << std::endl;\
    return EXIT_FAILURE;}} while(0)

constexpr int K = 10;
constexpr int D = 6;
constexpr int N = 50;
constexpr int M = 25;
constexpr int dT = 100; //100 ms? Idk, sure
constexpr int dWaypoint = dT/N; //amount of time per waypoint

__device__ int getTid() {
    return threadIdx.x + blockIdx.x*blockDim.x;
}

__device__ float getRandFloat(curandState& state) {
    return curand_uniform(&state) * FLT_MAX;
}

__global__ void optimize_trajectories(float* trajectories, float* noise_vectors, float* noisy_trajectories, curandState* states, unsigned int num_rngs_per_trajectory) {
    __shared__ float trajectory[N*D];
    __shared__ float noiseVectors[M][D];
    __shared__ float noisyTrajectories[M][N * D];
    int tid = getTid();
    //load RNG for the thread if it is supposed to have one. only max(N*D, M*D) threads are allocated an RNG.
    curandState local_state;
    if(threadIdx.x < num_rngs_per_trajectory)
        local_state = states[threadIdx.x + blockIdx.x * num_rngs_per_trajectory];

    //generate K random trajectories in parallel: the first ND threads generate the ND #s in this block's trajectory
    if(threadIdx.x < N*D) {
        float tmp = getRandFloat(local_state);
        //allow it to be readable by CPU, so write it to global mem as well as shared mem. TODO: consider delaying this write
        trajectories[tid] = tmp;
        trajectory[threadIdx.x] = tmp;
    }

    //generate M noise vectors for each trajectory
    if(threadIdx.x < M*D) {
        noiseVectors[threadIdx.x / D][threadIdx.x % D] = getRandFloat(local_state);
    }

    //generate the noisy trajectories
    if(threadIdx.x < M*N*D) {
        //TODO: this might be all threads in a block, in which case, this if doesn't need to be here
        int noisy_trajectory_index = threadIdx.x / N*D;
        //if threads in a warp access the same Smem location, it gets broadcast (as opposed to a bank conflict)
        //TODO: ensure these memory indicies are correct. I'm not certain they are- they won't segfault, but I don't think they will produce the correct result
        noisyTrajectories[noisy_trajectory_index][threadIdx.x % (N*D)] = noiseVectors[noisy_trajectory_index][threadIdx.x % D] + trajectory[threadIdx.x % (N*D)];
    }

    //do cost/collision detection

    //compute cost per noise vector

    //compute update vector

    //apply update vector to SMem trajectory

    //write back SMem trajectory

    //perform termination check. Might be a good do..while loop
}

__global__ void init_cudarand(curandState* states, unsigned int num_rngs) {
    unsigned int tid = getTid();
    if(tid < num_rngs) {
        //each thread gets the same seed (1234), a different sequence number, and no offset. This should be enough to ensure each RNG is sufficiently random
        curand_init(1234, tid, 0, states + tid);
    }
}

//TODO: integration as a ROSNode/MoveIt plugin :)
int main(int argc, char** argv) {
    int num_iterations = argc == 2 ? std::stoi(argv[1]) : 1;

    //initalization steps (set up cuRAND, allocate memory, etc)
    doapp::Scalar *device_trajectories;
    curandState *device_generators;
    //allocate memory
    int num_rngs = std::max(K*N*D, K*M*D);
    CUDA_CALL(cudaMalloc(&device_generators, num_rngs * sizeof(curandState)));

    //TODO: put these two calls on different streams to overlap memory alloc and rng initalization
    init_cudarand<<<ceil(double(num_rngs) / double(1024)), 1024>>>(device_generators, num_rngs);
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaMallocManaged(&device_trajectories, K*N*D*sizeof(doapp::Scalar)));


    //TODO: launch optimizing kernel num_iterations times
    for(int i = 0; i < num_iterations; ++i) {
	
    }

    cudaFree(device_trajectories);
    cudaFree(device_generators);
}
