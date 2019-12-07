#include "generate_trajectories.cuh"
#include <cfloat>
#include <cassert>

__device__ unsigned int getTid();
__device__ float getRandFloat(curandState& state);

__global__ void init_cudarand(curandState* states, unsigned int num_rngs) {
    unsigned int tid = getTid();
    if(tid < num_rngs) {
        //each thread gets the same seed (1234), a different sequence number, and no offset. This should be enough to ensure each RNG is sufficiently random
        curand_init(1234, tid, 0, states + tid);
    }
}

__device__ void initalize_trajectories(unsigned int num_waypoints, unsigned int waypoint_dim, curandState* rng, float* trajectories) {
    if(threadIdx.x < num_waypoints*waypoint_dim) {
            assert(rng);
            trajectories[threadIdx.x] = getRandFloat(*rng);
    }
}

__device__ void generate_noise_vectors(unsigned int num_noise_vectors, unsigned int noise_vector_dim, float* noise_vectors, curandState* rng) {
    if(threadIdx.x < num_noise_vectors*noise_vector_dim) {
        assert(rng);
        noise_vectors[threadIdx.x] = getRandFloat(*rng);
    }
}

__device__ void compute_noisy_trajectories(unsigned int num_noise_vectors, unsigned int dimensionality, unsigned int num_waypoints, float* noise_vectors, float* trajectories, float* noisy_trajectories) {
    if(threadIdx.x < num_waypoints*dimensionality*num_noise_vectors) {
        int noisy_trajectory_index = threadIdx.x / (num_waypoints*dimensionality);
        noisy_trajectories[threadIdx.x] = noise_vectors[noisy_trajectory_index * dimensionality + (threadIdx.x % dimensionality)] + trajectories[threadIdx.x % (num_waypoints * dimensionality)];
    }
}

__global__ void optimize_trajectories(float* trajectories, float* noise_vectors, float* noisy_trajectories, curandState* states, unsigned int num_rngs_per_trajectory, unsigned int num_waypoints, unsigned int waypoint_dim, unsigned int num_noise_vectors) {
    curandState* rng = threadIdx.x < num_rngs_per_trajectory ? states + (threadIdx.x + blockIdx.x*num_rngs_per_trajectory) : nullptr;
    //TODO: have a shared memory slice of everything read/written to for better access timing
    initalize_trajectories(num_waypoints, waypoint_dim, rng, trajectories);
    generate_noise_vectors(num_noise_vectors, waypoint_dim, noise_vectors, rng);
    compute_noisy_trajectories(num_noise_vectors, waypoint_dim, num_waypoints, noise_vectors, trajectories, noisy_trajectories);
}

__device__ unsigned int getTid() {
    return threadIdx.x + blockIdx.x*blockDim.x;
}

__device__ float getRandFloat(curandState& state) {
    return curand_uniform(&state) * FLT_MAX;
}
