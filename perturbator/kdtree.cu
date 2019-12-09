#include "kdtree.cuh"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

namespace doapp {
namespace kd_tree {

class BlockingWorkQueue {
public:
  void push(std::function<void()> &&f) {
    {
      const std::lock_guard<std::mutex> guard(mtx_);

      if (is_closed_) {
        throw std::runtime_error(
            "BlockingWorkQueue::push: cannot push onto closed queue");
      }

      functions_.push_back(std::move(f));
    }

    cv_.notify_one();
  }

  std::function<void()> pop() noexcept {
    std::unique_lock<std::mutex> guard(mtx_);

    while (!is_closed_ && functions_.empty()) {
      cv_.wait(guard);
    }

    if (functions_.empty()) {
      return {};
    }

    std::function<void()> popped = std::move(functions_.front());
    functions_.pop_front();

    return popped;
  }

  void close() noexcept {
    {
      const std::lock_guard<std::mutex> guard(mtx_);
      is_closed_ = true;
    }

    cv_.notify_all();
  }

private:
  std::deque<std::function<void()>> functions_;
  std::mutex mtx_;
  std::condition_variable cv_;
  bool is_closed_ = false;
};

enum class PartitionType {
  X,
  Y,
  Z,
};

enum class ChildType {
  Left,
  Right,
};

static void do_work(BlockingWorkQueue &queue) {
  std::function<void()> f = queue.pop();

  while (f) {
    f();

    f = queue.pop();
  }
}

static void build(BlockingWorkQueue &queue,
                  const Matrix<float, 3, Dynamic> &pointcloud,
                  Vector<std::size_t, Dynamic> &index_mapping,
                  Vector<kd_tree::Node, Dynamic> &nodes,
                  std::atomic<std::size_t> &nodes_top, std::size_t min_idx,
                  std::size_t max_idx, ChildType child_type,
                  PartitionType partition_type = PartitionType::X,
                  kd_tree::Node *parent = nullptr) noexcept;

static __host__ __device__ bool
do_neighbor_check(const Matrix<float, 3, Dynamic> &pointcloud,
                  const Vector<kd_tree::Node, Dynamic> &nodes, float x, float y,
                  float z, float radius_sq, float &current_min_sq,
                  std::size_t index = 0,
                  PartitionType partition_type = PartitionType::X) noexcept;

} // namespace kd_tree

KDTree::KDTree(const Matrix<float, 3, Dynamic> &pointcloud)
    : pointcloud_(pointcloud), nodes_(pointcloud.num_cols()) {
  if (pointcloud.num_cols() == 0) {
    return;
  }

  Vector<std::size_t, Dynamic> index_mapping(pointcloud.num_cols());
  std::iota(index_mapping.data(), index_mapping.data() + index_mapping.size(),
            static_cast<std::size_t>(0));
  std::atomic<std::size_t> nodes_top(0);

  kd_tree::BlockingWorkQueue queue;

#ifndef NDEBUG
  const std::size_t num_threads = 1;
#else
  const std::size_t num_threads = static_cast<std::size_t>(
      std::max(std::thread::hardware_concurrency(), 1u));
#endif

  queue.push([this, &index_mapping, &queue, &nodes_top] {
    kd_tree::build(queue, pointcloud_, index_mapping, nodes_, nodes_top, 0,
                   pointcloud_.num_cols(), kd_tree::ChildType::Left);
  });

  std::vector<std::thread> threads;
  for (std::size_t i = 0; i < num_threads; ++i) {
    threads.emplace_back([&queue] { kd_tree::do_work(queue); });
  }

  for (std::thread &t : threads) {
    t.join();
  }
}

__host__ __device__ bool
KDTree::has_neighbor_in_radius(float x, float y, float z, float radius) const
    noexcept {
  float current_min_sq = std::numeric_limits<float>::infinity();
  return kd_tree::do_neighbor_check(pointcloud_, nodes_, x, y, z,
                                    radius * radius, current_min_sq);
}

namespace kd_tree {

__host__ __device__ static constexpr float min(float x, float y) noexcept {
  return (y < x) ? y : x;
}

__host__ __device__ static constexpr float square(float x) noexcept {
  return x * x;
}

static void build(BlockingWorkQueue &queue,
                  const Matrix<float, 3, Dynamic> &pointcloud,
                  Vector<std::size_t, Dynamic> &index_mapping,
                  Vector<kd_tree::Node, Dynamic> &nodes,
                  std::atomic<std::size_t> &nodes_top, std::size_t min_idx,
                  std::size_t max_idx, ChildType child_type,
                  PartitionType partition_type,
                  kd_tree::Node *parent) noexcept {
  if (min_idx >= max_idx) {
    return;
  }

  const std::size_t n = max_idx - min_idx;
  const std::size_t mid_idx = min_idx + n / 2;
  const std::size_t top_idx = nodes_top.fetch_add(1, std::memory_order_relaxed);

  const auto min = index_mapping.data() + min_idx;
  const auto mid = index_mapping.data() + mid_idx;
  const auto max = index_mapping.data() + max_idx;

  float value;
  PartitionType next;
  kd_tree::Node &this_node = nodes[top_idx];

  switch (partition_type) {
  case PartitionType::X: {
    std::nth_element(min, mid, max,
                     [&pointcloud](std::size_t lhs, std::size_t rhs) {
                       return pointcloud[0][lhs] < pointcloud[0][rhs];
                     });
    value = pointcloud[0][*mid];

    next = PartitionType::Y;
  }
  case PartitionType::Y: {
    std::nth_element(min, mid, max,
                     [&pointcloud](std::size_t lhs, std::size_t rhs) {
                       return pointcloud[1][lhs] < pointcloud[1][rhs];
                     });
    value = pointcloud[1][*mid];

    next = PartitionType::Z;
  }
  case PartitionType::Z: {
    std::nth_element(min, mid, max,
                     [&pointcloud](std::size_t lhs, std::size_t rhs) {
                       return pointcloud[2][lhs] < pointcloud[2][rhs];
                     });
    value = pointcloud[2][*mid];

    next = PartitionType::X;
  }
  }

  const std::size_t pointcloud_index = *mid;
  this_node.value = value;
  this_node.pointcloud_index = pointcloud_index;

  if (parent) {
    switch (child_type) {
    case ChildType::Left:
      parent->left_child_offset = (&this_node - parent);

      break;
    case ChildType::Right:
      parent->right_child_offset = (&this_node - parent);

      break;
    }
  }

#ifndef NDEBUG
  constexpr std::size_t INLINE_THRESHOLD =
      std::numeric_limits<std::size_t>::max();
#else
  constexpr std::size_t INLINE_THRESHOLD = 1 << 10;
#endif

  if (mid_idx - min_idx > INLINE_THRESHOLD) {
    queue.push([&queue, &pointcloud, &index_mapping, &nodes, &nodes_top,
                min_idx, mid_idx, max_idx, next, &this_node] {
      build(queue, pointcloud, index_mapping, nodes, nodes_top, min_idx,
            mid_idx, ChildType::Left, next, &this_node);
    });
  } else {
    build(queue, pointcloud, index_mapping, nodes, nodes_top, min_idx, mid_idx,
          ChildType::Left, next, &this_node);
  }

  if (mid_idx + 1 - max_idx > INLINE_THRESHOLD) {
    queue.push([&queue, &pointcloud, &index_mapping, &nodes, &nodes_top,
                min_idx, mid_idx, max_idx, next, &this_node] {
      build(queue, pointcloud, index_mapping, nodes, nodes_top, mid_idx + 1,
            max_idx, ChildType::Right, next, &this_node);
    });
  } else {
    build(queue, pointcloud, index_mapping, nodes, nodes_top, mid_idx + 1,
          max_idx, ChildType::Right, next, &this_node);
  }

  if (top_idx + 1 == nodes.size()) {
    queue.close();
  }

#ifndef NDEBUG
  assert(this_node.pointcloud_index == pointcloud_index);
  assert(*mid == pointcloud_index);

  switch (partition_type) {
  case PartitionType::X: {
    assert(this_node.value = pointcloud[0][pointcloud_index]);

    break;
  }
  case PartitionType::Y: {
    assert(this_node.value = pointcloud[1][pointcloud_index]);

    break;
  }
  case PartitionType::Z: {
    assert(this_node.value = pointcloud[2][pointcloud_index]);

    break;
  }
  }
#endif
}

static __host__ __device__ bool
do_neighbor_check(const Matrix<float, 3, Dynamic> &pointcloud,
                  const Vector<kd_tree::Node, Dynamic> &nodes, float x, float y,
                  float z, float radius_sq, float &current_min_sq,
                  std::size_t index, PartitionType partition_type) noexcept {
  const kd_tree::Node &node = nodes[index];
  const std::size_t pointcloud_index = node.pointcloud_index;
  const float distance_to_node_sq =
      square(pointcloud[0][pointcloud_index] - x) +
      square(pointcloud[1][pointcloud_index] - y) +
      square(pointcloud[2][pointcloud_index] - z);

  current_min_sq = min(distance_to_node_sq, current_min_sq);

  if (current_min_sq <= radius_sq) {
    return true;
  }

  float distance_to_hyperplane_sq;
  PartitionType next_partition_type;

  std::size_t left_child_index = std::numeric_limits<std::size_t>::max();
  if (node.left_child_offset != std::numeric_limits<std::ptrdiff_t>::max()) {
    left_child_index = static_cast<std::size_t>(
        static_cast<std::ptrdiff_t>(index) + node.left_child_offset);
  }

  std::size_t right_child_index = std::numeric_limits<std::size_t>::max();
  if (node.right_child_offset != std::numeric_limits<std::ptrdiff_t>::max()) {
    right_child_index = static_cast<std::size_t>(
        static_cast<std::ptrdiff_t>(index) + node.right_child_offset);
  }

  std::size_t best_child_index;
  std::size_t other_child_index;

  switch (partition_type) {
  case PartitionType::X: {
    assert(node.value == pointcloud[0][pointcloud_index]);

    distance_to_hyperplane_sq = square(node.value - x);
    next_partition_type = PartitionType::Y;

    if (x <= node.value) {
      best_child_index = left_child_index;
      other_child_index = right_child_index;
    } else {
      best_child_index = right_child_index;
      other_child_index = left_child_index;
    }

    break;
  }
  case PartitionType::Y: {
    assert(node.value == pointcloud[1][pointcloud_index]);

    distance_to_hyperplane_sq = square(node.value - y);
    next_partition_type = PartitionType::Z;

    if (y <= node.value) {
      best_child_index = left_child_index;
      other_child_index = right_child_index;
    } else {
      best_child_index = right_child_index;
      other_child_index = left_child_index;
    }

    break;
  }

  case PartitionType::Z: {
    assert(node.value == pointcloud[2][pointcloud_index]);

    distance_to_hyperplane_sq = square(node.value - z);
    next_partition_type = PartitionType::X;

    if (z <= node.value) {
      best_child_index = left_child_index;
      other_child_index = right_child_index;
    } else {
      best_child_index = right_child_index;
      other_child_index = left_child_index;
    }

    break;
  }
  }

  if (best_child_index != std::numeric_limits<std::size_t>::max() &&
      do_neighbor_check(pointcloud, nodes, x, y, z, radius_sq, current_min_sq,
                        best_child_index, next_partition_type)) {
    return true;
  } else if (distance_to_hyperplane_sq <= current_min_sq &&
             other_child_index != std::numeric_limits<std::size_t>::max() &&
             do_neighbor_check(pointcloud, nodes, x, y, z, radius_sq,
                               current_min_sq, other_child_index,
                               next_partition_type)) {
    return true;
  }

  return false;
}

} // namespace kd_tree
} // namespace doapp
