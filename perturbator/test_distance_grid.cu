#include "distance_grid.cuh"
#include "matrix.cuh"
#include "random.cuh"

#include <chrono>
#include <iostream>
#include <vector>

#include <nanoflann.hpp>

using namespace doapp;

std::vector<float> random_pointcloud(std::size_t num_points) {
  std::uint64_t rng_state[2];
  doapp::detail::fill_with_randomness(rng_state, sizeof(rng_state));
  doapp::Pcg32 gen(rng_state[0], rng_state[1]);

  std::vector<float> v(num_points * 3);

  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i % 3 == 2) { // z
      v[i] = gen.rand_in_range(0.0f, 1.0f);
    } else {
      v[i] = gen.rand_in_range(-0.5f, 1.0f);
    }
  }

  return v;
}

std::vector<float> discretize(const std::vector<float> &pointcloud,
                              const distance_grid::Dimensions &dims) {
  assert(pointcloud.size() % 3 == 0);

  std::vector<std::uint8_t> occupied(dims.length * dims.width * dims.height,
                                     false);

  std::vector<float> discretized;
  discretized.reserve(dims.length * dims.width * dims.height * 3);

  const double x_offset =
      0.5 * static_cast<double>(dims.length) * dims.resolution;
  const double y_offset =
      0.5 * static_cast<double>(dims.width) * dims.resolution;

  for (std::size_t i = 0; i < pointcloud.size() / 3; ++i) {
    const double x = pointcloud[3 * i];
    const double y = pointcloud[3 * i + 1];
    const float z = pointcloud[3 * i + 2];

    const auto x_index =
        static_cast<std::size_t>((x + x_offset) / dims.resolution);
    const auto y_index =
        static_cast<std::size_t>((y + y_offset) / dims.resolution);
    const auto z_index = static_cast<std::size_t>(z / dims.resolution);

    const std::size_t flattened_index =
        dims.length * dims.width * z_index + dims.length * y_index + x_index;
    assert(flattened_index < occupied.size());

    if (!occupied[flattened_index]) {
      occupied[flattened_index] = true;
      discretized.insert(discretized.cend(), pointcloud.cbegin() + 3 * i,
                         pointcloud.cbegin() + 3 * i + 3);
    }
  }

  assert(discretized.size() % 3 == 0);

  return discretized;
}

int main() {
  constexpr std::size_t NUM_POINTS = 1 << 15;
  const auto pointcloud = random_pointcloud(NUM_POINTS);

  constexpr std::size_t LENGTH = 100; // 1m
  constexpr std::size_t WIDTH = 100;  // 1m
  constexpr std::size_t HEIGHT = 100; // 1m
  constexpr double RESOLUTION = 0.01; // 1cm

  constexpr distance_grid::Dimensions dims = {LENGTH, WIDTH, HEIGHT,
                                              RESOLUTION};

  auto start = std::chrono::steady_clock::now();
  const std::vector<float> discretized = discretize(pointcloud, dims);
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  std::cout << "discretization complete with " << discretized.size() / 3 << " points in " << elapsed.count() << "s\n";

  const distance_grid::VectorPointcloudAdaptor adaptor(pointcloud);
  distance_grid::KDTree tree(3, adaptor,
                             nanoflann::KDTreeSingleIndexAdaptorParams(16));
  tree.buildIndex();

  DistanceGrid grid(dims);

  start = std::chrono::steady_clock::now();
  grid.update(tree);
  end = std::chrono::steady_clock::now();
  elapsed = end - start;

  std::cout << "elapsed: " << elapsed.count() << "s\n";
}
