#ifndef KDTREE_CUH
#define KDTREE_CUH

#include "matrix.cuh"

#include <limits>

namespace doapp {
namespace kd_tree {

struct Node {
  float value;
  std::ptrdiff_t left_child_offset = std::numeric_limits<std::ptrdiff_t>::max();
  std::ptrdiff_t right_child_offset =
      std::numeric_limits<std::ptrdiff_t>::max();
  std::size_t pointcloud_index;
};

} // namespace kd_tree

class KDTree {
public:
  explicit KDTree(const Matrix<float, 3, Dynamic> &pointcloud);

  __host__ __device__ float distance_to_nearest_neighbor(float x, float y, float z) const noexcept;

private:
  const Matrix<float, 3, Dynamic> &pointcloud_;
  Vector<kd_tree::Node, Dynamic> nodes_;
};

} // namespace doapp

#endif
