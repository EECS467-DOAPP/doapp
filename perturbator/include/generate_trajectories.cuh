#include "common.cuh"
#include "cpu_distance_grid.cuh"

#include <cuda.h>
#include <curand_kernel.h>

__global__ void optimize_trajectories(
    float *trajectories, float *noise_vectors, float *noisy_trajectories,
    curandState *states, float *velocities, float *accelerations,
    float *smoothness, float *scores, float *update_vectors, bool *found_better,
    bool *time_expired, unsigned int num_rngs_per_trajectory,
    unsigned int num_waypoints, unsigned int waypoint_dim,
    unsigned int num_noise_vectors, float deltaT,
    const doapp::CPUDistanceGrid &grid);

__global__ void init_cudarand(curandState *states, unsigned int num_rngs);

__device__ void initalize_trajectories(unsigned int num_waypoints,
                                       unsigned int waypoint_dim,
                                       curandState *rng, float *trajectories);

__device__ void generate_noise_vectors(unsigned int num_noise_vectors,
                                       unsigned int noise_vector_dim,
                                       float *noise_vectors, curandState *rng);

__device__ void compute_noisy_trajectories(unsigned int num_noise_vectors,
                                           unsigned int dimensionality,
                                           unsigned int num_waypoints,
                                           float *noise_vectors,
                                           float *trajectories,
                                           float *noisy_trajectories);

__device__ void score_noisy_trajectories(
    float *noisy_trajectories, unsigned int num_noisy_trajectories,
    unsigned int num_waypoints, unsigned int waypoint_dim, float *scores,
    float *accelerations, float *smoothness, const doapp::CPUDistanceGrid &grid,
    volatile int *best_score);

__device__ void compute_update_vector(float *scores, float *noise_vectors,
                                      unsigned int num_noise_vectors,
                                      unsigned int waypoint_dim, float *output,
                                      float best_score);

__device__ unsigned int
num_collisions(float *waypoint, unsigned int waypoint_dim,
               const doapp::CPUDistanceGrid &grid) noexcept;

__device__ void compute_derivative(float *input_trajectories,
                                   unsigned int num_noisy_trajectories,
                                   unsigned int num_waypoints,
                                   unsigned int waypoint_dim, float deltaT,
                                   float *output);
__device__ void compute_acceleration(float *velocity,
                                     unsigned int num_noisy_trajectories,
                                     unsigned int num_waypoints,
                                     unsigned waypoint_dim, float deltaT,
                                     float *output);
__device__ void compute_velocity(float *input_trajectories,
                                 unsigned int num_noisy_trajectories,
                                 unsigned int num_waypoints,
                                 unsigned waypoint_dim, float deltaT,
                                 float *output);

extern __constant__ float initial_waypoint[];
extern __constant__ float final_waypoint[];
extern __constant__ float initial_trajectory[];
