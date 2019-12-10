#include "distance_grid.cuh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

#include <immintrin.h>

namespace doapp {
namespace distance_grid {

constexpr std::size_t AVX_ALIGN = 32;
constexpr std::size_t AVX_ALIGN_MASK = AVX_ALIGN - 1;

static constexpr std::size_t
divide_towards_positive_infinity(std::size_t numerator,
                                 std::size_t denominator) noexcept {
  return numerator / denominator +
         static_cast<std::size_t>(numerator % denominator != 0);
}

static short float_to_half_as_short(float x) noexcept {
  static_assert(sizeof(__half) == sizeof(short));

  const __half as_half = __float2half(x);

  short result;
  memcpy(&result, &as_half, sizeof(__half));

  return result;
}

} // namespace distance_grid

DistanceGrid::DistanceGrid(const distance_grid::Dimensions &dimensions)
    : dimensions_(dimensions) {
  const std::size_t vertical_slice_size =
      dimensions_.length * sizeof(__half) * dimensions_.width * sizeof(__half);

  // https://stackoverflow.com/questions/3407012/c-rounding-up-to-the-nearest-multiple-of-a-number
  const std::size_t slice_pitch_bytes =
      (vertical_slice_size + distance_grid::AVX_ALIGN - 1) &
      -distance_grid::AVX_ALIGN;
  assert(slice_pitch_bytes >= vertical_slice_size);
  assert(slice_pitch_bytes % distance_grid::AVX_ALIGN == 0);

  slice_pitch_ = slice_pitch_bytes / sizeof(__half); // from bytes to elements

  const std::size_t allocation_size = slice_pitch_bytes * dimensions_.height;
  const std::size_t aligned_allocation_size =
      allocation_size + distance_grid::AVX_ALIGN;

  void *ptr;
  if (cudaMallocManaged(&ptr, aligned_allocation_size) != cudaSuccess) {
    throw std::runtime_error("DistanceGrid::DistanceGrid: cudaMallocManaged");
  }

  base_ = static_cast<__half *>(ptr);

  if (reinterpret_cast<std::uintptr_t>(ptr) % distance_grid::AVX_ALIGN != 0) {
    aligned_base_ = reinterpret_cast<__half *>(
        (reinterpret_cast<std::uintptr_t>(ptr) &
         ~static_cast<std::uintptr_t>(distance_grid::AVX_ALIGN_MASK)) +
        distance_grid::AVX_ALIGN);
  } else {
    aligned_base_ = base_;
  }

  assert(reinterpret_cast<std::uintptr_t>(aligned_base_) %
             distance_grid::AVX_ALIGN ==
         0);

  x_offset_ = static_cast<float>(0.5 * static_cast<double>(dimensions_.length) *
                                 dimensions_.resolution);
  y_offset_ = static_cast<float>(0.5 * static_cast<double>(dimensions_.width) *
                                 dimensions_.resolution);
}

DistanceGrid::~DistanceGrid() { cudaFree(base_); }

void DistanceGrid::update(const distance_grid::KDTree &tree) noexcept {
  const auto num_threads = static_cast<std::size_t>(
      std::max(1u, std::thread::hardware_concurrency()));

  const std::size_t slices_per_thread =
      distance_grid::divide_towards_positive_infinity(dimensions_.height,
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

__host__ __device__ const __half &DistanceGrid::operator()(float x, float y,
                                                           float z) const
    noexcept {
  const auto x_index = static_cast<std::size_t>(x + x_offset_);
  const auto y_index = static_cast<std::size_t>(y + y_offset_);
  const auto z_index = static_cast<std::size_t>(z);

  return aligned_base_[slice_pitch_ * z_index + dimensions_.length * y_index +
                       x_index];
}

void DistanceGrid::update_thread(const distance_grid::KDTree &tree,
                                 std::size_t min_height,
                                 std::size_t max_height) noexcept {
  const std::size_t slice_numel = dimensions_.width * dimensions_.height;
  const double x_min =
      0.5 * dimensions_.resolution / static_cast<double>(dimensions_.length);
  const double y_min =
      0.5 * dimensions_.resolution / static_cast<double>(dimensions_.length);

  std::size_t x = 0;
  std::size_t y = 0;
  float x_as_float;
  float y_as_float;

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

    x_as_float = static_cast<float>(x_min + static_cast<double>(x) /
                                                dimensions_.resolution);
    y_as_float = static_cast<float>(y_min + static_cast<double>(y) /
                                                dimensions_.resolution);
  };

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
    __half *const slice_base_ptr = aligned_base_ + slice_pitch_ * z;
    auto write_ptr = reinterpret_cast<__m256i *>(slice_base_ptr);
    const auto z_as_float =
        static_cast<float>(dimensions_.resolution * static_cast<double>(z));

    short elems[16];
    std::size_t n = (slice_numel + 15) / 16;

    // https://en.wikipedia.org/wiki/Duff%27s_device
    switch (slice_numel % 16) {
    case 0:
      do {
        elems[0] = distance_grid::float_to_half_as_short(
            get_distance(x_as_float, y_as_float, z_as_float));
        increment_xy();
      case 15:
        elems[1] = distance_grid::float_to_half_as_short(
            get_distance(x_as_float, y_as_float, z_as_float));
        increment_xy();
      case 14:
        elems[2] = distance_grid::float_to_half_as_short(
            get_distance(x_as_float, y_as_float, z_as_float));
        increment_xy();
      case 13:
        elems[3] = distance_grid::float_to_half_as_short(
            get_distance(x_as_float, y_as_float, z_as_float));
        increment_xy();
      case 12:
        elems[4] = distance_grid::float_to_half_as_short(
            get_distance(x_as_float, y_as_float, z_as_float));
        increment_xy();
      case 11:
        elems[5] = distance_grid::float_to_half_as_short(
            get_distance(x_as_float, y_as_float, z_as_float));
        increment_xy();
      case 10:
        elems[6] = distance_grid::float_to_half_as_short(
            get_distance(x_as_float, y_as_float, z_as_float));
        increment_xy();
      case 9:
        elems[7] = distance_grid::float_to_half_as_short(
            get_distance(x_as_float, y_as_float, z_as_float));
        increment_xy();
      case 8:
        elems[8] = distance_grid::float_to_half_as_short(
            get_distance(x_as_float, y_as_float, z_as_float));
        increment_xy();
      case 7:
        elems[9] = distance_grid::float_to_half_as_short(
            get_distance(x_as_float, y_as_float, z_as_float));
        increment_xy();
      case 6:
        elems[10] = distance_grid::float_to_half_as_short(
            get_distance(x_as_float, y_as_float, z_as_float));
        increment_xy();
      case 5:
        elems[11] = distance_grid::float_to_half_as_short(
            get_distance(x_as_float, y_as_float, z_as_float));
        increment_xy();
      case 4:
        elems[12] = distance_grid::float_to_half_as_short(
            get_distance(x_as_float, y_as_float, z_as_float));
        increment_xy();
      case 3:
        elems[13] = distance_grid::float_to_half_as_short(
            get_distance(x_as_float, y_as_float, z_as_float));
        increment_xy();
      case 2:
        elems[14] = distance_grid::float_to_half_as_short(
            get_distance(x_as_float, y_as_float, z_as_float));
        increment_xy();
      case 1:
        elems[15] = distance_grid::float_to_half_as_short(
            get_distance(x_as_float, y_as_float, z_as_float));
        increment_xy();

        const __m256i elems_vec = _mm256_set_epi16(
            elems[15], elems[14], elems[13], elems[12], elems[11], elems[10],
            elems[9], elems[8], elems[7], elems[6], elems[5], elems[4],
            elems[3], elems[2], elems[1], elems[0]);
        _mm256_store_si256(write_ptr, elems_vec);
        ++write_ptr;
      } while (--n > 0);
    }
  }
}

} // namespace doapp
