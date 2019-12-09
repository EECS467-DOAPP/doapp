#include "kdtree.cuh"
#include "matrix.cuh"
#include "random.cuh"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>

#include <sys/random.h>

int main() {
  constexpr std::size_t N = 1 << 20;

  std::uint64_t rng_state[2];
  doapp::detail::fill_with_randomness(rng_state, sizeof(rng_state));
  doapp::Pcg32 gen(rng_state[0], rng_state[1]);

  doapp::Matrix<float, 3, doapp::Dynamic> pointcloud(3, N);

  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      pointcloud[i][j] = gen.rand_in_range(-10.0f, 20.0f);
    }
  }

  const auto start = std::chrono::steady_clock::now();
  doapp::KDTree tree(pointcloud);
  const auto end = std::chrono::steady_clock::now();

  const std::chrono::duration<double> elapsed = end - start;

  std::cout << elapsed.count() << "s\n";
}
