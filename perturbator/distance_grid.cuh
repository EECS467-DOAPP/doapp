#ifndef DISTANCE_GRID_CUH
#define DISTANCE_GRID_CUH

#include <cstddef>
#include <vector>

#include <cuda_fp16.h>

#include <nanoflann.hpp>

namespace doapp {
namespace distance_grid {

struct Dimensions {
  std::size_t length; // x
  std::size_t width;  // y
  std::size_t height; // z
  double resolution;   // meters
};

class VectorPointcloudAdaptor {
public:
    using coord_t = float;

    explicit VectorPointcloudAdaptor(const std::vector<float> &v) noexcept
        : vector_ptr_(&v) {
          assert(v.size() % 3 == 0);
        }

    std::size_t kdtree_get_point_count() const noexcept {
        assert(vector_ptr_);

        return static_cast<std::size_t>(vector_ptr_->size() / 3);
    }

    float kdtree_get_pt(size_t index, size_t dimension) const {
        assert(vector_ptr_);
        assert(index < vector_ptr_->size() / 3);
        assert(dimension < 3);

        return (*vector_ptr_)[index * 3 + dimension];
    }

    template <typename BoundingBox>
    bool kdtree_get_bbox(BoundingBox &) const noexcept {
        return false;
    }

private:
    const std::vector<float> *vector_ptr_;
};

using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, VectorPointcloudAdaptor>,
    VectorPointcloudAdaptor, 3>;

} // namespace distance_grid

class DistanceGrid {
public:
  DistanceGrid(const distance_grid::Dimensions &dimensions);
  DistanceGrid(DistanceGrid &&other) noexcept = delete;

  ~DistanceGrid();

  DistanceGrid &operator=(DistanceGrid &&other) noexcept = delete;

  void update(const distance_grid::KDTree &tree) noexcept;
  __host__ __device__ const __half &operator()(float x, float y, float z) const noexcept;

private:
  void update_thread(const distance_grid::KDTree &tree, std::size_t min_height, std::size_t max_height) noexcept;

  __half *base_ = nullptr;
  __half *aligned_base_ = nullptr;

  distance_grid::Dimensions dimensions_;
  std::size_t slice_pitch_;
  float x_offset_;
  float y_offset_;
};

} // namespace doapp

#endif
