#include "cpu_distance_grid.cuh"
#include "distance_grid.cuh"
#include "matrix.cuh"
#include "random.cuh"

#include <chrono>
#include <cstdio>

using namespace doapp;

Matrix<float, Dynamic, 3> random_pointcloud(std::size_t num_points) {
  doapp::Pcg32 gen;

  Matrix<float, Dynamic, 3> A(num_points, 3);

  for (std::size_t i = 0; i < A.num_rows(); ++i) {
    A[i][0] = gen.rand_in_range(-0.5f, 1.0f);
    A[i][1] = gen.rand_in_range(-0.5f, 1.0f);
    A[i][2] = gen.rand_in_range(0.0f, 1.0f);
  }

  return A;
}

int main() {
  constexpr std::size_t NUM_POINTS = 1 << 15;
  const auto pointcloud = random_pointcloud(NUM_POINTS);

  constexpr std::size_t LENGTH = 100; // 1m
  constexpr std::size_t WIDTH = 100;  // 1m
  constexpr std::size_t HEIGHT = 100; // 1m
  constexpr double RESOLUTION = 0.01; // 10cm

  const cpu_distance_grid::MatrixPointcloudAdaptor adaptor(pointcloud);
  cpu_distance_grid::KDTree tree(3, adaptor,
                                 nanoflann::KDTreeSingleIndexAdaptorParams(16));
  tree.buildIndex();

  constexpr cpu_distance_grid::Dimensions cpu_dims = {LENGTH, WIDTH, HEIGHT,
                                                      RESOLUTION};

  CPUDistanceGrid cpu_grid(cpu_dims);

  const auto start = std::chrono::steady_clock::now();
  cpu_grid.update(tree);
  const auto end = std::chrono::steady_clock::now();
  const std::chrono::duration<double> elapsed = end - start;
  printf("cpu elapsed: %fs\n", elapsed.count());
}
