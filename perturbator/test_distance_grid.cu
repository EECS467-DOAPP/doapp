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

int main() {
  constexpr std::size_t NUM_POINTS = 1 << 16;
  const auto pointcloud = random_pointcloud(NUM_POINTS);

  const distance_grid::VectorPointcloudAdaptor adaptor(pointcloud);
  distance_grid::KDTree tree(3, adaptor,
                             nanoflann::KDTreeSingleIndexAdaptorParams(16));
  tree.buildIndex();

  constexpr std::size_t LENGTH = 100; // 1m
  constexpr std::size_t WIDTH = 100;  // 1m
  constexpr std::size_t HEIGHT = 100; // 1m
  constexpr double RESOLUTION = 0.01; // 1cm

  constexpr distance_grid::Dimensions dims = {LENGTH, WIDTH, HEIGHT,
                                              RESOLUTION};

  DistanceGrid grid(dims);

  const auto start = std::chrono::steady_clock::now();
  grid.update(tree);
  const auto end = std::chrono::steady_clock::now();
  const std::chrono::duration<double> elapsed = end - start;

  std::cout << "elapsed: " << elapsed.count() << "s\n";
}
