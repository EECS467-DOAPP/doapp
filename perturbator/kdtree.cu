#include "kdtree.cuh"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace doapp {

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

struct XComparator {
  bool operator()(Slice<float> lhs, Slice<float> rhs) const noexcept {
    return lhs[0] < rhs[0];
  }
};

struct YComparator {
  bool operator()(Slice<float> lhs, Slice<float> rhs) const noexcept {
    return lhs[1] < rhs[1];
  }
};

struct ZComparator {
  bool operator()(Slice<float> lhs, Slice<float> rhs) const noexcept {
    return lhs[2] < rhs[2];
  }
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
                  Matrix<float, 3, Dynamic> &pointcloud,
                  Vector<kd_tree::Node, Dynamic> &nodes,
                  std::atomic<std::size_t> &nodes_top, std::size_t min_idx,
                  std::size_t max_idx, ChildType child_type,
                  PartitionType partition_type = PartitionType::X,
                  kd_tree::Node *parent = nullptr) noexcept;

KDTree::KDTree(Matrix<float, 3, Dynamic> &pointcloud)
    : pointcloud_(pointcloud), nodes_(pointcloud.num_cols()) {
  if (pointcloud.num_cols() == 0) {
    return;
  }

  std::atomic<std::size_t> nodes_top(0);

  BlockingWorkQueue queue;

  const std::size_t num_threads =
      std::max(std::thread::hardware_concurrency(), 1u);

  queue.push([this, &queue, &nodes_top] {
    build(queue, pointcloud_, nodes_, nodes_top, 0, pointcloud_.num_cols(),
          ChildType::Left);
  });

  std::vector<std::thread> threads;
  for (std::size_t i = 0; i < num_threads; ++i) {
    threads.emplace_back([&queue] { do_work(queue); });
  }

  for (std::thread &t : threads) {
    t.join();
  }
}

__host__ __device__ float KDTree::nearest_neighbor_distance(float x, float y,
                                                            float z) const
    noexcept {
  return -1.0f;
}

static void build(BlockingWorkQueue &queue,
                  Matrix<float, 3, Dynamic> &pointcloud,
                  Vector<kd_tree::Node, Dynamic> &nodes,
                  std::atomic<std::size_t> &nodes_top, std::size_t min_idx,
                  std::size_t max_idx, ChildType child_type,
                  PartitionType partition_type,
                  kd_tree::Node *parent) noexcept {
  if (min_idx >= max_idx) {
    return;
  }

  MatrixColIterator<float, 3, Dynamic> first(pointcloud, min_idx);
  MatrixColIterator<float, 3, Dynamic> last(pointcloud, max_idx);

  const std::size_t n = max_idx - min_idx;
  const std::size_t mid_idx = min_idx + n / 2;

  MatrixColIterator<float, 3, Dynamic> mid(pointcloud, mid_idx);

  const std::size_t top_idx = nodes_top.fetch_add(1, std::memory_order_relaxed);

  float value;
  PartitionType next;
  kd_tree::Node &this_node = nodes[top_idx];

  switch (partition_type) {
  case PartitionType::X: {
    std::nth_element(first, mid, last, XComparator());
    value = (*mid)[0];

    next = PartitionType::Y;
  }
  case PartitionType::Y: {
    std::nth_element(first, mid, last, YComparator());
    value = (*mid)[1];

    next = PartitionType::Z;
  }
  case PartitionType::Z: {
    std::nth_element(first, mid, last, ZComparator());
    value = (*mid)[2];

    next = PartitionType::X;
  }
  }

  this_node.value = value;
  this_node.pointcloud_index = mid_idx;

  if (parent) {
    switch (child_type) {
    case ChildType::Left:
      parent->left_child_offset = (parent - &this_node);

      break;
    case ChildType::Right:
      parent->right_child_offset = (parent - &this_node);

      break;
    }
  }

  constexpr std::size_t INLINE_THRESHOLD = 1 << 8;

  if (mid_idx - min_idx > INLINE_THRESHOLD) {
    queue.push([&queue, &pointcloud, &nodes, &nodes_top, min_idx, mid_idx,
                max_idx, next, &this_node] {
      build(queue, pointcloud, nodes, nodes_top, min_idx, mid_idx,
            ChildType::Left, next, &this_node);
    });
  } else {
    build(queue, pointcloud, nodes, nodes_top, min_idx, mid_idx,
          ChildType::Left, next, &this_node);
  }

  if (mid_idx + 1 - max_idx > INLINE_THRESHOLD) {
    queue.push([&queue, &pointcloud, &nodes, &nodes_top, min_idx, mid_idx,
                max_idx, next, &this_node] {
      build(queue, pointcloud, nodes, nodes_top, mid_idx + 1, max_idx,
            ChildType::Right, next, &this_node);
    });
  } else {
    build(queue, pointcloud, nodes, nodes_top, mid_idx + 1, max_idx,
          ChildType::Right, next, &this_node);
  }

  if (top_idx + 1 == nodes.size()) {
    queue.close();
  }
}

} // namespace doapp
