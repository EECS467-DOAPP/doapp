#include "cpu_distance_grid.cuh"
#include "distance_grid.cuh"
#include "matrix.cuh"
#include "random.cuh"

#include <chrono>
#include <iostream>

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

  constexpr distance_grid::Dimensions dims = {LENGTH, WIDTH, HEIGHT,
                                              RESOLUTION};

  DistanceGrid grid(dims);

  auto start = std::chrono::steady_clock::now();
  grid.update(pointcloud);
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "gpu elapsed: " << elapsed.count() << "s\n";

  const cpu_distance_grid::MatrixPointcloudAdaptor adaptor(pointcloud);
  cpu_distance_grid::KDTree tree(3, adaptor,
                                 nanoflann::KDTreeSingleIndexAdaptorParams(16));
  tree.buildIndex();

  constexpr cpu_distance_grid::Dimensions cpu_dims = {LENGTH, WIDTH, HEIGHT,
                                                      RESOLUTION};

  CPUDistanceGrid cpu_grid(cpu_dims);

  start = std::chrono::steady_clock::now();
  cpu_grid.update(tree);
  end = std::chrono::steady_clock::now();
  elapsed = end - start;
  std::cout << "cpu elapsed: " << elapsed.count() << "s\n";

  for (std::size_t z = 0; z < HEIGHT; ++z) {
    const float z_f = static_cast<float>(z) * RESOLUTION;

    for (std::size_t y = 0; y < WIDTH; ++y) {
      const float y_f = static_cast<float>(y) * RESOLUTION - 0.5f;

      for (std::size_t x = 0; x < LENGTH; ++x) {
        const float x_f = static_cast<float>(x) * RESOLUTION - 0.5f;

        std::cout << "grid(" << x << ", " << y << ", " << z
                  << ") = " << grid(x_f, y_f, z_f) << ", cpu_grid(" << x << ", "
                  << y << ", " << z << ") = " << cpu_grid(x_f, y_f, z_f)
                  << '\n';
      }
    }
  }
}
