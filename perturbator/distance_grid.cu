#include "distance_grid.cuh"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <iostream>

#include <immintrin.h>

namespace doapp {
namespace distance_grid {

__host__ __device__ static std::uint32_t as_u32(float f32) noexcept {
  static_assert(sizeof(float) == sizeof(std::uint32_t));

  union Converter {
    float f;
    std::uint32_t u;
  };

  Converter c = {f32};

  return c.u;
}

__host__ __device__ static float as_f32(std::uint32_t u32) noexcept {
 static_assert(sizeof(float) == sizeof(std::uint32_t));

  union Converter {
    std::uint32_t u;
    float f;
  };

  Converter c = {u32};

  return c.f;
}

__host__ __device__ static constexpr std::uint32_t
divide_towards_positive_infinity(std::uint32_t numerator,
                                 std::uint32_t denominator) noexcept {
  return numerator / denominator +
         static_cast<std::uint32_t>(numerator % denominator != 0);
}

template <typename T>
__host__ __device__ static constexpr T min(T x, T y) noexcept {
  return (y < x) ? y : x;
}

template <typename T>
__host__ __device__ static constexpr T square(T x) noexcept {
  return x * x;
}

__device__ void atomic_min(volatile float *memory, float value) noexcept {
  float current_f32 = *memory;
  unsigned current = as_u32(current_f32);

  while (!(current_f32 <= value)) {
    current =
        atomicCAS(reinterpret_cast<unsigned *>(const_cast<float *>(memory)),
                  current, as_u32(value));
    current_f32 = as_f32(current);
  }
}

struct float3 {
  float x;
  float y;
  float z;
};

constexpr std::size_t SHARED_MEMORY_SIZE = 16 * (1 << 10);
constexpr std::uint32_t POINTCLOUD_PARTITION_MAX_ALLOCATION =
    SHARED_MEMORY_SIZE / 4;
constexpr std::uint32_t POINT_SIZE = sizeof(float) * 3; // x y z

constexpr std::uint32_t GRID_MAX_ALLOCATION =
    SHARED_MEMORY_SIZE - POINTCLOUD_PARTITION_MAX_ALLOCATION;
constexpr std::uint32_t MAX_CELLS_PER_BLOCK =
    GRID_MAX_ALLOCATION / sizeof(float);

constexpr std::uint32_t MAX_POINTS_PER_POINTCLOUD_PARTITION =
    POINTCLOUD_PARTITION_MAX_ALLOCATION / POINT_SIZE;

__global__ static void
update_grid(Dimensions dimensions, const float3 *pointcloud, float *output,
            std::uint32_t num_points,
            std::uint32_t points_per_pointcloud_partition,
            std::uint32_t cells_per_cell_partition,
            std::uint32_t points_to_copy_per_thread,
            std::uint32_t cells_to_write_per_thread) {
  // copy a partition of the pointcloud from global to shared memory
  __shared__ float result_grid[MAX_CELLS_PER_BLOCK];
  float *const result_grid_end = result_grid + cells_per_cell_partition;

  __shared__ float3 pointcloud_partition[MAX_POINTS_PER_POINTCLOUD_PARTITION];
  float3 *const pointcloud_partition_end =
      pointcloud_partition + points_per_pointcloud_partition;

  // blockIdx.x: pointcloud partition index
  // blockIdx.y: cell partition index

  const float3 *const global_partition_begin =
      pointcloud + points_per_pointcloud_partition * blockIdx.x;
  const float3 *const global_partition_end =
      min(global_partition_begin + points_per_pointcloud_partition,
          pointcloud + num_points);

  const float3 *const this_thread_global_partition_begin =
      global_partition_begin + points_to_copy_per_thread * threadIdx.x;
  const float3 *const this_thread_global_partition_end =
      min(this_thread_global_partition_begin + points_to_copy_per_thread,
          global_partition_end);

  float3 *this_thread_shared_partition_begin =
      pointcloud_partition + points_to_copy_per_thread * threadIdx.x;
  float3 *this_thread_shared_partition_end =
      min(this_thread_shared_partition_begin + points_to_copy_per_thread,
          pointcloud_partition + points_per_pointcloud_partition);

  const float3 *global_read_head = this_thread_global_partition_begin;
  float3 *shared_write_head = this_thread_shared_partition_begin;

  for (; global_read_head < this_thread_global_partition_end &&
         shared_write_head < this_thread_shared_partition_end;
       ++global_read_head, ++shared_write_head) {
    *shared_write_head = *global_read_head;
  }

  __syncthreads();
  // shared memory is filled in; compute distance grid for a subset of points

  const std::uint32_t num_cells =
      dimensions.length * dimensions.width * dimensions.height;

  const float x_offset =
      -0.5f * static_cast<float>(dimensions.length) * dimensions.resolution;
  const float y_offset =
      -0.5f * static_cast<float>(dimensions.width) * dimensions.resolution;

  const std::uint32_t block_cell_index_begin =
      cells_per_cell_partition * blockIdx.y;
  const std::uint32_t block_cell_index_end =
      min(block_cell_index_begin + cells_per_cell_partition, num_cells);

  const std::uint32_t thread_cell_index_begin =
      block_cell_index_begin + cells_to_write_per_thread * threadIdx.x;
  const std::uint32_t thread_cell_index_end =
      min(thread_cell_index_begin + cells_to_write_per_thread,
          block_cell_index_end);

  for (std::uint32_t i = thread_cell_index_begin; i < thread_cell_index_end;
       ++i) {
    const std::uint32_t z_index = i / (dimensions.length * dimensions.width);
    const std::uint32_t slice_index =
        i % (dimensions.length * dimensions.width);

    const std::uint32_t y_index = slice_index / dimensions.length;
    const std::uint32_t x_index = slice_index % dimensions.length;

    const float z = static_cast<float>(z_index) * dimensions.resolution;
    const float y =
        static_cast<float>(y_index) * dimensions.resolution + y_offset;
    const float x =
        static_cast<float>(x_index) * dimensions.resolution + x_offset;

    float cell_value_sq = HUGE_VALF;

    for (float3 *point = pointcloud_partition; point < pointcloud_partition_end;
         ++point) {
      const float new_distance_sq =
          square(x - point->x) + square(y - point->y) + square(z - point->z);
      cell_value_sq = min(cell_value_sq, new_distance_sq);
    }

    result_grid[i - block_cell_index_begin] = std::sqrt(cell_value_sq);
  }

  float *const output_end =
      output + dimensions.length * dimensions.width * dimensions.height;
  float *const cell_output_begin =
      output + cells_per_cell_partition * blockIdx.y;
  float *const cell_output_end =
      min(cell_output_begin + cells_per_cell_partition, output_end);

  float *const this_thread_cell_begin =
      cell_output_begin + cells_to_write_per_thread * threadIdx.x;
  float *const this_thread_cell_end =
      min(this_thread_cell_begin + cells_to_write_per_thread, cell_output_end);

  const float *const this_thread_shared_cell_begin =
      result_grid + cells_to_write_per_thread * threadIdx.x;
  const float *const this_thread_shared_cell_end =
      min(this_thread_shared_cell_begin + cells_to_write_per_thread,
          static_cast<const float *>(result_grid_end));

  volatile float *write_head = this_thread_cell_begin;
  const float *read_head = this_thread_shared_cell_begin;

  for (; write_head < this_thread_cell_end &&
         read_head < this_thread_shared_cell_end;
       ++write_head, ++read_head) {
    atomic_min(write_head, *read_head);
  }
}

} // namespace distance_grid

DistanceGrid::DistanceGrid(const distance_grid::Dimensions &dimensions) noexcept
    : distances_(dimensions.length * dimensions.width * dimensions.height),
      dimensions_(dimensions) {
  std::fill(distances_.data(), distances_.data() + distances_.size(),
            HUGE_VALF);

  x_offset_ = static_cast<float>(0.5 * static_cast<double>(dimensions_.length) *
                                 static_cast<double>(dimensions_.resolution));
  y_offset_ = static_cast<float>(0.5 * static_cast<double>(dimensions_.width) *
                                 static_cast<double>(dimensions_.resolution));

  x_min_ = -x_offset_;
  x_max_ = x_offset_;

  y_min_ = -y_offset_;
  y_max_ = y_offset_;

  z_max_ = static_cast<float>(static_cast<double>(dimensions_.height) *
                              static_cast<double>(dimensions_.resolution));

  slice_pitch_ = dimensions_.length * dimensions_.height;
}

void DistanceGrid::update(const Matrix<float, Dynamic, 3> &pointcloud) {
  std::fill(distances_.data(), distances_.data() + distances_.size(),
            HUGE_VALF);
  const std::uint32_t num_pointcloud_partitions =
      distance_grid::divide_towards_positive_infinity(
          static_cast<std::uint32_t>(pointcloud.num_rows()),
          distance_grid::MAX_POINTS_PER_POINTCLOUD_PARTITION);
  const std::uint32_t points_per_pointcloud_partition =
      distance_grid::divide_towards_positive_infinity(
          static_cast<std::uint32_t>(pointcloud.num_rows()),
          num_pointcloud_partitions);

  const std::uint32_t num_cells =
      dimensions_.length * dimensions_.width * dimensions_.height;
  const std::uint32_t num_cell_partitions =
      distance_grid::divide_towards_positive_infinity(
          num_cells, distance_grid::MAX_CELLS_PER_BLOCK);
  const std::uint32_t cells_per_cell_partition =
      distance_grid::divide_towards_positive_infinity(num_cells,
                                                      num_cell_partitions);

  constexpr std::uint32_t BLOCK_SIZE = 1024;

  const std::uint32_t points_to_copy_per_thread =
      distance_grid::divide_towards_positive_infinity(
          points_per_pointcloud_partition, BLOCK_SIZE);
  const std::uint32_t cells_to_write_per_thread =
      distance_grid::divide_towards_positive_infinity(cells_per_cell_partition,
                                                      BLOCK_SIZE);

  const auto pointcloud_ptr =
      reinterpret_cast<const distance_grid::float3 *>(pointcloud.data());
  float *const output = distances_.data();
  const auto num_points = static_cast<std::uint32_t>(pointcloud.num_rows());

  distance_grid::
      update_grid<<<dim3(num_pointcloud_partitions, num_cell_partitions, 1),
                    BLOCK_SIZE, distance_grid::SHARED_MEMORY_SIZE>>>(
          dimensions_, pointcloud_ptr, output, num_points,
          points_per_pointcloud_partition, cells_per_cell_partition,
          points_to_copy_per_thread, cells_to_write_per_thread);

  if (cudaDeviceSynchronize() != cudaSuccess) {
    throw std::runtime_error("doapp::DistanceGrid::DistanceGrid: update");
  }
}

__host__ __device__ float DistanceGrid::operator()(float x, float y,
                                                   float z) const noexcept {
  if (!(x >= x_min_ && x <= x_max_) || !(y >= y_min_ && y <= y_max_) ||
      !(z >= 0.0 && z <= z_max_)) {
    return HUGE_VALF;
  }

  const auto x_index =
      static_cast<std::uint32_t>((x + x_offset_) / dimensions_.resolution);
  const auto y_index =
      static_cast<std::uint32_t>((y + y_offset_) / dimensions_.resolution);
  const auto z_index = static_cast<std::uint32_t>(z / dimensions_.resolution);

  return distances_[slice_pitch_ * z_index + dimensions_.length * y_index +
                    x_index];
}

} // namespace doapp
