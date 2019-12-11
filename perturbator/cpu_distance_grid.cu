#include "cpu_distance_grid.cuh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

#include <iostream>

#include <immintrin.h>

namespace doapp {
namespace cpu_distance_grid {

constexpr std::size_t AVX_ALIGN = 32;
constexpr std::size_t AVX_ALIGN_MASK = AVX_ALIGN - 1;

static constexpr std::size_t
divide_towards_positive_infinity(std::size_t numerator,
                                 std::size_t denominator) noexcept {
  return numerator / denominator +
         static_cast<std::size_t>(numerator % denominator != 0);
}

} // namespace cpu_distance_grid

CPUDistanceGrid::CPUDistanceGrid(
    const cpu_distance_grid::Dimensions &dimensions)
    : dimensions_(dimensions) {
  const std::size_t vertical_slice_size =
      dimensions_.length * sizeof(float) * dimensions_.width * sizeof(float);

  // https://stackoverflow.com/questions/3407012/c-rounding-up-to-the-nearest-multiple-of-a-number
  const std::size_t slice_pitch_bytes =
      (vertical_slice_size + cpu_distance_grid::AVX_ALIGN - 1) &
      -cpu_distance_grid::AVX_ALIGN;
  assert(slice_pitch_bytes >= vertical_slice_size);
  assert(slice_pitch_bytes % cpu_distance_grid::AVX_ALIGN == 0);

  slice_pitch_ = slice_pitch_bytes / sizeof(float); // from bytes to elements

  const std::size_t allocation_size = slice_pitch_bytes * dimensions_.height;
  const std::size_t aligned_allocation_size =
      allocation_size + cpu_distance_grid::AVX_ALIGN;

  void *ptr;
  if (cudaMallocManaged(&ptr, aligned_allocation_size) != cudaSuccess) {
    throw std::runtime_error(
        "CPUDistanceGrid::CPUDistanceGrid: cudaMallocManaged");
  }

  base_ = static_cast<float *>(ptr);

  if (reinterpret_cast<std::uintptr_t>(ptr) % cpu_distance_grid::AVX_ALIGN !=
      0) {
    aligned_base_ = reinterpret_cast<float *>(
        (reinterpret_cast<std::uintptr_t>(ptr) &
         ~static_cast<std::uintptr_t>(cpu_distance_grid::AVX_ALIGN_MASK)) +
        cpu_distance_grid::AVX_ALIGN);
  } else {
    aligned_base_ = base_;
  }

  assert(reinterpret_cast<std::uintptr_t>(aligned_base_) %
             cpu_distance_grid::AVX_ALIGN ==
         0);

  resolution_ = static_cast<float>(dimensions_.resolution);
  x_offset_ = static_cast<float>(0.5 * static_cast<double>(dimensions_.length) *
                                 dimensions_.resolution);
  y_offset_ = static_cast<float>(0.5 * static_cast<double>(dimensions_.width) *
                                 dimensions_.resolution);

  x_min_ = -x_offset_;
  x_max_ = x_offset_;

  y_min_ = -y_offset_;
  y_max_ = y_offset_;

  z_max_ = static_cast<float>(static_cast<double>(dimensions_.height) *
                              static_cast<double>(dimensions_.resolution));
}

CPUDistanceGrid::~CPUDistanceGrid() { cudaFree(base_); }

void CPUDistanceGrid::update(const cpu_distance_grid::KDTree &tree) noexcept {
  const auto num_threads = static_cast<std::size_t>(
      std::max(1u, std::thread::hardware_concurrency()));

  const std::size_t slices_per_thread =
      cpu_distance_grid::divide_towards_positive_infinity(dimensions_.height,
                                                          num_threads);

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (std::size_t i = 0; i < num_threads; ++i) {
    const std::size_t min_height = i * slices_per_thread;
    const std::size_t max_height =
        std::min((i + 1) * slices_per_thread, dimensions_.height);

    threads.emplace_back([this, &tree, min_height, max_height] {
      update_thread(tree, min_height, max_height);
    });
  }

  for (std::thread &t : threads) {
    t.join();
  }
}

__host__ __device__ float CPUDistanceGrid::operator()(float x, float y,
                                                      float z) const noexcept {
  if (!(x >= x_min_ && x <= x_max_) || !(y >= y_min_ && y <= y_max_) ||
      !(z >= 0.0 && z <= z_max_)) {
    return HUGE_VALF;
  }

  const auto x_index = static_cast<std::size_t>((x + x_offset_) / resolution_);
  const auto y_index = static_cast<std::size_t>((y + y_offset_) / resolution_);
  const auto z_index = static_cast<std::size_t>(z / resolution_);

  return aligned_base_[slice_pitch_ * z_index + dimensions_.length * y_index +
                       x_index];
}

void CPUDistanceGrid::update_thread(const cpu_distance_grid::KDTree &tree,
                                    std::size_t min_height,
                                    std::size_t max_height) noexcept {
  const std::size_t slice_numel = dimensions_.width * dimensions_.height;
  const double x_min =
      0.5 * dimensions_.resolution / static_cast<double>(dimensions_.length);
  const double y_min =
      0.5 * dimensions_.resolution / static_cast<double>(dimensions_.length);

  const auto get_distance = [&tree](float x, float y, float z) {
    std::size_t return_index;
    float out_distance_squared;
    nanoflann::KNNResultSet<float> result_set(1);
    result_set.init(&return_index, &out_distance_squared);

    const float query_point[3] = {x, y, z};
    if (tree.findNeighbors(result_set, query_point,
                           nanoflann::SearchParams())) {
      return std::sqrt(out_distance_squared);
    } else {
      return std::numeric_limits<float>::infinity();
    }
  };

  for (std::size_t z = min_height; z < max_height; ++z) {
    float *const slice_base_ptr = aligned_base_ + slice_pitch_ * z;
    const auto z_as_float =
        static_cast<float>(dimensions_.resolution * static_cast<double>(z));

    std::size_t x = 0;
    std::size_t y = 0;
    float x_as_float = static_cast<float>(x_min);
    float y_as_float = static_cast<float>(y_min);

    const auto increment_xy = [this, x_min, y_min, &x, &y, &x_as_float,
                               &y_as_float] {
      ++x;

      if (x >= dimensions_.length) {
        x = 0;
        ++y;

        if (y >= dimensions_.height) {
          y = 0;
        }
      }

      x_as_float = static_cast<float>(x_min + static_cast<double>(x) *
                                                  dimensions_.resolution);
      y_as_float = static_cast<float>(y_min + static_cast<double>(y) *
                                                  dimensions_.resolution);
    };

    std::size_t slice_index = 0;

    for (; slice_index < slice_numel; slice_index += 8) {
      float elems[8] = {};

      std::size_t to_fill = 8;
      if (slice_numel - slice_index < 8) {
        to_fill -= slice_numel % 8;
      }

      for (std::size_t i = 0; i < to_fill; ++i) {
        elems[i] = get_distance(x_as_float, y_as_float, z_as_float);
        increment_xy();
      }

      const __m256 elems_vec =
          _mm256_set_ps(elems[7], elems[6], elems[5], elems[4], elems[3],
                        elems[2], elems[1], elems[0]);
      _mm256_store_ps(slice_base_ptr + slice_index, elems_vec);
    }
  }
}

} // namespace doapp
