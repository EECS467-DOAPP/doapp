#ifndef KDTREE_CUH
#define KDTREE_CUH

#include "matrix.cuh"

#include <limits>

namespace doapp {

class KDTree {
public:
    explicit KDTree(Matrix<float, 3, Dynamic> &pointcloud);

    __host__ __device__ float nearest_neighbor_distance(float x, float y, float z) noexcept;

private:
  struct Node {
    explicit Node(float v) noexcept : value(v) {}

    float value;
    std::size_t left_child = std::numeric_limits<std::size_t>::max();
    std::size_t right_child = std::numeric_limits<std::size_t>::max();
  };

  Matrix<float, 3, Dynamic> &pointcloud_;
  Vector<Node, Dynamic> nodes_;
};

} // namespace doapp

#endif
