#ifndef TRAJECTORY_CUH
#define TRAJECTORY_CUH

#include "common.cuh"
#include "vector.cuh"

#include <cstddef>

namespace doapp {

template <std::size_t Dimension> class Trajectory {
public:
  explicit Trajectory(std::size_t n) : points_(n) {}

  __host__ __device__ Vector<Scalar, static_cast<std::ptrdiff_t>(Dimension)> &
  operator[](std::size_t i) noexcept {
    assert(i < points_.size());

    return points_[i];
  }

  __host__ __device__ const
      Vector<Scalar, static_cast<std::ptrdiff_t>(Dimension)> &
      operator[](std::size_t i) const noexcept {
    assert(i < points_.size());

    return points_[i];
  }

private:
  Vector<Vector<Scalar, static_cast<std::ptrdiff_t>(Dimension)>, Dynamic>
      points_;
};

} // namespace doapp

#endif
