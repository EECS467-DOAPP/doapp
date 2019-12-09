#include "kdtree.cuh"
#include "matrix.cuh"
#include "random.cuh"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>

#include <sys/random.h>

constexpr float square(float x) noexcept { return x * x; }

int main() {
  constexpr std::size_t N = 1 << 15;

  std::uint64_t rng_state[2];
  doapp::detail::fill_with_randomness(rng_state, sizeof(rng_state));
  doapp::Pcg32 gen(rng_state[0], rng_state[1]);

  doapp::Matrix<float, 3, doapp::Dynamic> pointcloud(3, N);

  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      pointcloud[i][j] = gen.rand_in_range(-5.0f, 10.0f);
    }
  }

  auto start = std::chrono::steady_clock::now();
  doapp::KDTree tree(pointcloud);
  auto end = std::chrono::steady_clock::now();

  std::chrono::duration<double> elapsed = end - start;

  std::cout << elapsed.count() << "s\n";

  const float x = gen.rand_in_range(-5.0f, 10.0f);
  const float y = gen.rand_in_range(-5.0f, 10.0f);
  const float z = gen.rand_in_range(-5.0f, 10.0f);

  std::cout << "x: " << x << ", y: " << y << ", z: " << z << '\n';

  start = std::chrono::steady_clock::now();
  float min_dist_sq = std::numeric_limits<float>::infinity();
  for (std::size_t i = 0; i < N; ++i) {
    const float dist_sq = square(x - pointcloud[0][i]) +
                          square(y - pointcloud[1][i]) +
                          square(z - pointcloud[2][i]);

    min_dist_sq = std::min(dist_sq, min_dist_sq);
  }

  const float min_dist = std::sqrt(min_dist_sq);

  end = std::chrono::steady_clock::now();
  elapsed = end - start;

  std::cout << elapsed.count() << "s, min_dist: " << min_dist << '\n';

  start = std::chrono::steady_clock::now();
  const bool has_neighbor =
      tree.has_neighbor_in_radius(x, y, z, min_dist * 1.001f);
  end = std::chrono::steady_clock::now();
  elapsed = end - start;
  std::cout << elapsed.count() << "s, has_neighbor: " << std::boolalpha
            << has_neighbor << '\n';
}
