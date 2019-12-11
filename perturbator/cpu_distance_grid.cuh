#ifndef CPU_DISTANCE_GRID_CUH
#define CPU_DISTANCE_GRID_CUH

#include <cstddef>

#include "matrix.cuh"

#include <cuda_fp16.h>

#include <nanoflann.hpp>

namespace doapp {
namespace cpu_distance_grid {

struct Dimensions {
  std::size_t length; // x
  std::size_t width;  // y
  std::size_t height; // z
  double resolution;  // meters
};

class MatrixPointcloudAdaptor {
public:
  using coord_t = float;

  explicit MatrixPointcloudAdaptor(const Matrix<float, Dynamic, 3> &m) noexcept
      : matrix_ptr_(&m) {
  }

  std::size_t kdtree_get_point_count() const noexcept {
    assert(matrix_ptr_);

    return matrix_ptr_->num_rows();
  }

  float kdtree_get_pt(size_t index, size_t dimension) const {
    assert(matrix_ptr_);
    assert(index < matrix_ptr_->num_rows());
    assert(dimension < matrix_ptr_->num_cols());

    return (*matrix_ptr_)[index][dimension];
  }

  template <typename BoundingBox>
  bool kdtree_get_bbox(BoundingBox &) const noexcept {
    return false;
  }

private:
  const Matrix<float, Dynamic, 3> *matrix_ptr_;
};

using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, MatrixPointcloudAdaptor>,
    MatrixPointcloudAdaptor, 3>;

} // namespace cpu_distance_grid

class CPUDistanceGrid {
public:
  CPUDistanceGrid(const cpu_distance_grid::Dimensions &dimensions);
  CPUDistanceGrid(CPUDistanceGrid &&other) noexcept = delete;

  ~CPUDistanceGrid();

  CPUDistanceGrid &operator=(CPUDistanceGrid &&other) noexcept = delete;

  void update(const cpu_distance_grid::KDTree &tree) noexcept;
  __host__ __device__ float operator()(float x, float y, float z) const
      noexcept;

private:
  void update_thread(const cpu_distance_grid::KDTree &tree,
                     std::size_t min_height, std::size_t max_height) noexcept;

  float *base_ = nullptr;
  float *aligned_base_ = nullptr;

  cpu_distance_grid::Dimensions dimensions_;
  std::size_t slice_pitch_;
  float resolution_;
  float x_offset_;
  float y_offset_;

  float x_min_;
  float x_max_;
  float y_min_;
  float y_max_;
  float z_max_;
};

} // namespace doapp

#endif
