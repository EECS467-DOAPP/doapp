#include <cuda.h>
#include <curand_kernel.h>

__global__ void optimize_trajectories(float* trajectories, float* noise_vectors, float* noisy_trajectories, curandState* states, unsigned int num_rngs_per_trajectory, unsigned int num_waypoints, unsigned int waypoint_dim, unsigned int num_noise_vectors);

__global__ void init_cudarand(curandState* states, unsigned int num_rngs);

__device__ void initalize_trajectories(unsigned int num_waypoints, unsigned int waypoint_dim, curandState* rng, float* trajectories);

__device__ void generate_noise_vectors(unsigned int num_noise_vectors, unsigned int noise_vector_dim, float* noise_vectors, curandState* rng);

__device__ void compute_noisy_trajectories(unsigned int num_noise_vectors, unsigned int dimensionality, unsigned int num_waypoints, float* noise_vectors, float* trajectories, float* noisy_trajectories);
