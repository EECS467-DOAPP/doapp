#include "planner.cuh"

#include "random.cuh"
#include "unique_ptr.cuh"

namespace doapp {
namespace detail {

// k, m, n, and d are implied based on grid and block dimension
__global__ static void do_plan_kernel();

void do_plan(std::size_t k, std::size_t m, std::size_t n, std::size_t d,
             float *output, const float *initial_position,
             const float *initial_velocity,
             const JointPositionBound *position_bounds,
             const JointVelocityBound *velocity_bounds,
             const JointAccelerationBound *acceleration_bounds,
             std::chrono::steady_clock::duration timeout);

} // namespace detail
} // namespace doapp
