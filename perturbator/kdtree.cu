#include "kdtree.cuh"

#include <algorithm>

namespace doapp {

struct XComparator {
  bool operator()(Slice<const float> lhs, Slice<const float> rhs) const
      noexcept {
    return lhs[0] < rhs[0];
  }
};

struct YComparator {
  bool operator()(Slice<const float> lhs, Slice<const float> rhs) const
      noexcept {
    return lhs[1] < rhs[1];
  }
};

struct ZComparator {
  bool operator()(Slice<const float> lhs, Slice<const float> rhs) const
      noexcept {
    return lhs[2] < rhs[2];
  }
};

enum class PartitionType {
  X,
  Y,
  Z,
};

static void build(Matrix<float, 3, Dynamic> &pointcloud,
                  Vector<Node, Dynamic> &nodes, std::size_t &nodes_top,
                  std::size_t min_idx, std::size_t max_idx,
                  PartitionType partition_type = PartitionType::X) noexcept;

KDTree::KDTree(Matrix<float, 3, Dynamic> &pointcloud)
    : nodes_(pointcloud.num_cols()) {
  std::size_t nodes_top = 0;
  build(pointcloud, nodes_, nodes_top, 0, pointcloud.num_cols());
}

static void build(Matrix<float, 3, Dynamic> &pointcloud,
                  Vector<Node, Dynamic> &nodes, std::size_t &nodes_top,
                  std::size_t min_idx, std::size_t max_idx,
                  PartitionType partition_type = PartitionType::X) noexcept {
  MatrixColIterator<float, 3, Dynamic> first(pointcloud, min_idx);
  MatrixColIterator<float, 3, Dynamic> last(pointcloud, max_idx);

  switch (partition_type) {
    case PartitionType::X: {
      nodes[nodes_top].value
    }
    case PartitionType::Y: {

    }
    case PartitionType::Z: {

    }
  }

  ++nodes_top;
}

} // namespace doapp
