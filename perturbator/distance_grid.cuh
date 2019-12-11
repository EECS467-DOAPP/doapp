#ifndef DISTANCE_GRID_CUH
#define DISTANCE_GRID_CUH

#include <cstdint>

#include "matrix.cuh"
#include "vector.cuh"

namespace doapp {
namespace distance_grid {

struct Dimensions {
  std::uint32_t length; // x
  std::uint32_t width;  // y
  std::uint32_t height; // z
  float resolution;   // meters
};

} // namespace distance_grid

class DistanceGrid {
public:
  DistanceGrid(const distance_grid::Dimensions &dimensions) noexcept;

  void update(const Matrix<float, Dynamic, 3> &pointcloud);
  __host__ __device__ float operator()(float x, float y, float z) const noexcept;

private:
  Vector<float, Dynamic> distances_;

  distance_grid::Dimensions dimensions_;

  float x_offset_;
  float y_offset_;
  float x_min_;
  float x_max_;
  float y_min_;
  float y_max_;
  float z_max_;

  std::uint32_t slice_pitch_;
};

} // namespace doapp

#endif
