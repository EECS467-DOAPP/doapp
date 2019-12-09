#include "kdtree.cuh"
#include "matrix.cuh"
#include "random.cuh"
#include "unique_ptr.cuh"

#include <chrono>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <ratio>
#include <vector>

#include <sys/random.h>

__host__ __device__ constexpr float square(float x) noexcept { return x * x; }
__host__ __device__ constexpr float gpumin(float x, float y) noexcept {
  return (y < x) ? y : x;
}

__global__ void
brute_force_search(doapp::Slice<const float> query_points,
                   doapp::Slice<float> min_distances,
                   const doapp::Matrix<float, 3, doapp::Dynamic> &pointcloud) {
  const std::size_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_index >= min_distances.size()) {
    return;
  }

  const std::size_t query_offset = thread_index * 3;
  const float x = query_points[query_offset];
  const float y = query_points[query_offset + 1];
  const float z = query_points[query_offset + 2];

  float min_dist_sq = HUGE_VAL;
  for (std::size_t i = 0; i < pointcloud.num_cols(); ++i) {
    const float dist_sq = square(x - pointcloud[0][i]) +
                          square(y - pointcloud[1][i]) +
                          square(z - pointcloud[2][i]);

    min_dist_sq = gpumin(dist_sq, min_dist_sq);
  }

  min_distances[thread_index] = std::sqrt(min_dist_sq);
}

__global__ void kdtree_search(doapp::Slice<const float> query_points,
                              doapp::Slice<float> min_distances,
                              const doapp::KDTree &tree) {
  const std::size_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_index >= min_distances.size()) {
    return;
  }

  const std::size_t query_offset = thread_index * 3;

  min_distances[thread_index] = tree.distance_to_nearest_neighbor(
      query_points[query_offset], query_points[query_offset + 1],
      query_points[query_offset + 2]);
}

struct Event {
  Event() {
    if (cudaEventCreate(&event) != cudaSuccess) {
      throw std::runtime_error("Event::Event: cudaEventCreate");
    }
  }

  ~Event() { cudaEventDestroy(event); }

  void record() const {
    if (cudaEventRecord(event) != cudaSuccess) {
      throw std::runtime_error("Event::record: cudaEventRecord");
    }
  }

  void synchronize() const {
    if (cudaEventSynchronize(event) != cudaSuccess) {
      throw std::runtime_error("Event::synchronize: cudaEventSynchronize");
    }
  }

  cudaEvent_t event;
};

std::chrono::duration<double> elapsed_time(const Event &start,
                                           const Event &end) {
  float elapsed_ms;

  if (cudaEventElapsedTime(&elapsed_ms, start.event, end.event) !=
      cudaSuccess) {
    throw std::runtime_error("elapsed_time: cudaEventElapsedTime");
  }

  return std::chrono::duration<float, std::milli>(elapsed_ms);
}

__host__ __device__ constexpr std::size_t
divide_towards_positive_infinity(std::size_t numerator,
                                 std::size_t denominator) noexcept {
  return numerator / denominator +
         static_cast<std::size_t>(numerator % denominator != 0);
}

int main() {
  constexpr std::size_t N = 1 << 15;

  std::uint64_t rng_state[2];
  doapp::detail::fill_with_randomness(rng_state, sizeof(rng_state));
  doapp::Pcg32 gen(rng_state[0], rng_state[1]);

  const auto pointcloud =
      doapp::make_unique<doapp::Matrix<float, 3, doapp::Dynamic>>(3, N);

  for (std::size_t i = 0; i < pointcloud->num_rows(); ++i) {
    for (std::size_t j = 0; j < pointcloud->num_cols(); ++j) {
      (*pointcloud)[i][j] = gen.rand_in_range(-5.0f, 10.0f);
    }
  }

  const auto tree = doapp::make_unique<doapp::KDTree>(*pointcloud);

  constexpr std::size_t K = 4096;

  doapp::Vector<float, doapp::Dynamic> query_points(K * 3);
  std::generate(query_points.data(), query_points.data() + query_points.size(),
                [&gen] { return gen.rand_in_range(-5.0f, 10.0f); });
  doapp::Vector<float, doapp::Dynamic> nearest_distances(K);

  const auto &q = query_points;

  static constexpr std::size_t BLOCK_SIZE = 1024;
  static constexpr std::size_t NUM_BLOCKS =
      divide_towards_positive_infinity(K, BLOCK_SIZE);

  std::chrono::duration<double> elapsed;

  {
    Event start;
    Event stop;

    start.record();
    brute_force_search<<<NUM_BLOCKS, BLOCK_SIZE>>>(
        q.as_slice(), nearest_distances.as_slice(), *pointcloud);
    stop.record();

    stop.synchronize();
    elapsed = elapsed_time(start, stop);
  }

  std::cout << "brute force GPU: " << elapsed.count() << "s\n";

  {
    Event start;
    Event stop;

    start.record();
    kdtree_search<<<NUM_BLOCKS, BLOCK_SIZE>>>(
        q.as_slice(), nearest_distances.as_slice(), *tree);
    stop.record();

    stop.synchronize();
    elapsed = elapsed_time(start, stop);
  }

  std::cout << "k-d tree GPU: " << elapsed.count() << "s\n";
}
