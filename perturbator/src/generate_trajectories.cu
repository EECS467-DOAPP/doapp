#include "generate_trajectories.cuh"
#include "common.cuh"
#include <cassert>
#include <cstdio>

__constant__ float initial_waypoint[doapp::num_joints];
__constant__ float final_waypoint[doapp::num_joints];
__constant__ float initial_trajectory[doapp::num_joints * 50];
__device__ unsigned int getTid();
__device__ float getRandFloat(curandState* state, float min, float max);
__device__ float getRandFloat(curandState* state, float min, float max, float score);

__global__ void init_cudarand(curandState* states, unsigned int num_rngs) {
    unsigned int tid = getTid();
    if(tid < num_rngs) {
        //each thread gets the same seed (1234), a different sequence number, and no offset. This should be enough to ensure each RNG is sufficiently random
        curand_init(1234, tid, 0, states + tid);
    }
}

__device__ void initalize_trajectories(unsigned int num_waypoints, unsigned int waypoint_dim, curandState* rng, float* trajectories) {
    // random trajectories
    if(threadIdx.x < num_waypoints*waypoint_dim) {
            assert(rng);
            float range = doapp::max_joint_angles[threadIdx.x % waypoint_dim] - doapp::min_joint_angles[threadIdx.x % waypoint_dim];
            //min: min - max
            //max: max - min
            trajectories[threadIdx.x] = getRandFloat(rng, -range / 10, range / 10);
    }
}

__device__ void generate_noise_vectors(unsigned int num_noise_vectors, unsigned int noise_vector_dim, float* noise_vectors, curandState* rng) {
    if(threadIdx.x < num_noise_vectors*noise_vector_dim) {
        assert(rng);
        noise_vectors[threadIdx.x] = getRandFloat(rng, doapp::min_joint_angles[threadIdx.x % noise_vector_dim], doapp::max_joint_angles[threadIdx.x % noise_vector_dim]);
    }
}

__device__ void compute_noisy_trajectories(unsigned int num_noise_vectors, unsigned int dimensionality, unsigned int num_waypoints, float* noise_vectors, float* trajectories, float* noisy_trajectories) {
    if(threadIdx.x < num_waypoints*dimensionality*num_noise_vectors) {
        int noisy_trajectory_index = threadIdx.x / (num_waypoints*dimensionality);
        noisy_trajectories[threadIdx.x] = noise_vectors[noisy_trajectory_index * dimensionality + (threadIdx.x % dimensionality)] + trajectories[threadIdx.x % (num_waypoints * dimensionality)];
    }
}

__device__ void score_noisy_trajectories(float* noisy_trajectories, unsigned int num_noisy_trajectories, unsigned int num_waypoints, unsigned int waypoint_dim, float* scores, float* accelerations, float* smoothness, volatile int* best_score) {
    //initalize values
    if(threadIdx.x < num_noisy_trajectories) {
        scores[threadIdx.x] = 0;
        smoothness[threadIdx.x] = 0;
    }
    __syncthreads();
    //compute sum of squared accelerations
    if(threadIdx.x < num_noisy_trajectories*num_waypoints*waypoint_dim) {
        unsigned int trajectory = threadIdx.x / (num_waypoints*waypoint_dim);
        atomicAdd(smoothness + trajectory, accelerations[threadIdx.x] * accelerations[threadIdx.x]);
    }
    __syncthreads();
    //multiply by factor of 1/2
    if(threadIdx.x < num_noisy_trajectories) {
        smoothness[threadIdx.x] *= 0.5f;
    }
    __syncthreads();
    //compute cost due to collisions
    float waypoint[doapp::num_joints];
    if(threadIdx.x < num_noisy_trajectories*num_waypoints) {
        //load waypoint. TODO: first transpose noisy_trajectories such that accesses are coalesced. This should result in a decent speedup
        unsigned int trajectory = threadIdx.x / num_waypoints;
        unsigned int waypoint_index = threadIdx.x % num_waypoints;
        for(unsigned int i = 0; i < waypoint_dim; ++i) {
            waypoint[i] = noisy_trajectories[trajectory*num_waypoints*waypoint_dim + waypoint_index*waypoint_dim + i]; //currently, memory accesses between threads are strided by waypoint_dim (AKA not coalesced), and that is not good
        }
        float waypoint_cost = num_collisions(waypoint, doapp::num_joints);
        atomicAdd(scores + trajectory, waypoint_cost); //could do a list reduction, but like, this is one line that does the same thing
    }
    __syncthreads();
    //add smoothness cost
    if(threadIdx.x < num_noisy_trajectories) {
        smoothness[threadIdx.x] *= 0.5f;
        scores[threadIdx.x] += smoothness[threadIdx.x];
        int rounded_score = int(ceilf(scores[threadIdx.x]));
        //printf("thread %d has score %d\n", threadIdx.x, rounded_score);
        atomicMin((int*)best_score, rounded_score);
    }
    //TODO: put in a term about violating joint limits. be sure to move the atomicMin line!
}

__device__ void compute_update_vector(float* scores, float* noise_vectors, unsigned int num_noise_vectors, unsigned int waypoint_dim, float* output, float best_score) {
    if(threadIdx.x < waypoint_dim) {
        output[threadIdx.x] = 0;
    }
    __syncthreads(); //TODO: look into thread groups cause not all threads in the whole block need to sync up here
    if(threadIdx.x < num_noise_vectors*waypoint_dim) {
        unsigned int noise_vector = threadIdx.x / waypoint_dim;
        float our_score = scores[noise_vector];
        float weight = best_score / ceilf(our_score);
        atomicAdd(output + (threadIdx.x % waypoint_dim), weight * noise_vectors[threadIdx.x]);
    }
}

__device__ void update_trajectories(float* trajectory, float* update_vector, unsigned int num_waypoints, unsigned int waypoint_dim) {
    if(threadIdx.x < num_waypoints*waypoint_dim) {
        trajectory[threadIdx.x] -= 10000.0f * update_vector[threadIdx.x % waypoint_dim];
    }
}

__global__ void optimize_trajectories(float* trajectories, float* noise_vectors, float* noisy_trajectories, curandState* states, float* velocities, float* accelerations, float* smoothness, float* scores, float* update_vectors, bool* found_better, bool* time_expired, unsigned int num_rngs_per_trajectory, unsigned int num_waypoints, unsigned int waypoint_dim, unsigned int num_noise_vectors, float deltaT) {
    curandState* rng = threadIdx.x < num_rngs_per_trajectory ? states + (threadIdx.x + blockIdx.x*num_rngs_per_trajectory) : nullptr;
    float* block_trajectories = trajectories + blockIdx.x*num_waypoints*waypoint_dim;
    float* block_noise_vectors = noise_vectors + blockIdx.x*num_noise_vectors*waypoint_dim;
    float* block_noisy_trajectories = noisy_trajectories + blockIdx.x*num_noise_vectors*num_waypoints*waypoint_dim;
    float* block_velocities = velocities + blockIdx.x*num_noise_vectors*num_waypoints*waypoint_dim;
    float* block_accelerations = accelerations + blockIdx.x*num_noise_vectors*num_waypoints*waypoint_dim;
    float* block_smoothness = smoothness + blockIdx.x*num_noise_vectors;
    float* block_scores = scores + blockIdx.x*(num_noise_vectors + 1) + 1;
    float* block_update_vector = update_vectors + blockIdx.x*waypoint_dim;
    __shared__ volatile int best_score, current_score, best_trajectory_index;
    if(!threadIdx.x) {
        best_score = INT_MAX; //ensure someone sets it
        current_score = INT_MAX;
        best_trajectory_index = -1;
    }

    //TODO: have a shared memory slice of everything read/written to for better access timing
    //initalize_trajectories(num_waypoints, waypoint_dim, rng, block_trajectories);
    //first, score our inital trajectory
    compute_velocity(initial_trajectory, 1, num_waypoints, waypoint_dim, deltaT, block_velocities);
    __syncthreads();
    compute_acceleration(initial_trajectory, 1, num_waypoints, waypoint_dim, deltaT, block_accelerations);
    //TODO: do a list reduction on accelerations (all threads in a block participate) but instead of just adding them, square individual terms first
    __syncthreads();
    //score initial trajectory into block_scores[-1]
    score_noisy_trajectories(initial_trajectory, 1, num_waypoints, waypoint_dim, block_scores - 1, block_accelerations, block_smoothness, &current_score);
    __syncthreads();
    if(!threadIdx.x && !blockIdx.x) {
        printf("inital trajectory score: %d\n", current_score);
    }
    bool use_inital_trajectory = true;
    do {
        //attempt to optimize it
        generate_noise_vectors(num_noise_vectors, waypoint_dim, block_noise_vectors, rng);
        __syncthreads();
        compute_noisy_trajectories(num_noise_vectors, waypoint_dim, num_waypoints, block_noise_vectors, use_inital_trajectory ? initial_trajectory : block_trajectories, block_noisy_trajectories);
        __syncthreads();
        compute_velocity(block_noisy_trajectories, num_noise_vectors, num_waypoints, waypoint_dim, deltaT, block_velocities);
        __syncthreads();
        compute_acceleration(block_velocities, num_noise_vectors, num_waypoints, waypoint_dim, deltaT, block_accelerations);
        //TODO: do a list reduction on accelerations (all threads in a block participate) but instead of just adding them, square individual terms first
        __syncthreads();
        if(!threadIdx.x)
            best_score = INT_MAX; //ensure someone sets it
        __syncthreads();
        score_noisy_trajectories(block_noisy_trajectories, num_noise_vectors, num_waypoints, waypoint_dim, block_scores, block_accelerations, block_smoothness, &best_score);
        __syncthreads();
        //keep the best trajectories
        if(best_score < current_score) {
            if(threadIdx.x < num_noise_vectors*num_waypoints*waypoint_dim) {
                int trajectory = threadIdx.x / (num_waypoints * waypoint_dim);
                if(int(ceilf(scores[trajectory])) == best_score) {
                    block_trajectories[threadIdx.x] = block_noisy_trajectories[threadIdx.x];
                }
            }
            __syncthreads();
            if(!threadIdx.x) {
                best_score = current_score;
                use_inital_trajectory = false;
                printf("New best score for block %d is %d\n", blockIdx.x, best_score);
            }
        } else if(!threadIdx.x) {
            //printf("Block %d did not find a better trajectory\n", blockIdx.x);
        }
        /*
        __syncthreads();
        //TODO: consider also scoring the current trajectory to make sure we don't go in a worse direction
        compute_update_vector(block_scores, block_noise_vectors, num_noise_vectors, waypoint_dim, block_update_vector, best_score);
        __syncthreads();
        update_trajectories(block_trajectories, block_update_vector, num_waypoints, waypoint_dim);
        */
    //} while(++count < 5);
    } while(!*time_expired);
    if(!threadIdx.x) {
        printf("Found better: %s\n", use_inital_trajectory ? "false" : "true");
        found_better[blockIdx.x] = !use_inital_trajectory;
    }
}

__device__ void compute_acceleration(float* velocity, unsigned int num_noisy_trajectories, unsigned int num_waypoints, unsigned waypoint_dim, float deltaT, float* output) {
    compute_derivative(velocity, num_noisy_trajectories, num_waypoints, waypoint_dim, deltaT, output);
}
__device__ void compute_velocity(float* input_trajectories, unsigned int num_noisy_trajectories, unsigned int num_waypoints, unsigned waypoint_dim, float deltaT, float* output) {
    compute_derivative(input_trajectories, num_noisy_trajectories, num_waypoints, waypoint_dim, deltaT, output);
}
__device__ void compute_derivative(float* input, unsigned int num_noisy_trajectories, unsigned int num_waypoints, unsigned int waypoint_dim, float deltaT, float* output) {
    if(threadIdx.x < num_noisy_trajectories*num_waypoints*waypoint_dim) {
        unsigned int waypoint = (threadIdx.x / waypoint_dim) % num_waypoints;
        unsigned int dim = threadIdx.x % waypoint_dim;
        float prior_val, current_val;
        prior_val = waypoint ? input[threadIdx.x - waypoint_dim] : initial_waypoint[dim];
        current_val = input[threadIdx.x];
        float val = (current_val - prior_val) / deltaT;
        output[threadIdx.x] = val;
    }
}

__device__ unsigned int getTid() {
    return threadIdx.x + blockIdx.x*blockDim.x;
}

/*
__device__ float getRandFloat(curandState* state, float min, float max) {
    return curand_normal(state) * (max - min + 0.999999) + min;
}
*/

__device__ float getRandFloat(curandState* state, float min_val, float max_val) {
    return curand_normal(state);
    //return min(max(curand_normal(state) * 0.01f, min_val), max_val);
}
__device__ unsigned int num_collisions(float* waypoint, unsigned int waypoint_dim) {
    return 0; //TODO: use Greg's impl. For testing, might want to see if we can inverse the order of the trajectories based upon this or something idk
}
