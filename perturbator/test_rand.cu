#include "random.cuh"
#include "vector.cuh"

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

using Pcg32Vec = doapp::Vector<doapp::Pcg32, doapp::Dynamic>;
using FloatVec = doapp::Vector<float, doapp::Dynamic>;
using FloatVecVec = doapp::Vector<FloatVec, doapp::Dynamic>;

__global__ void generate_many_random(doapp::Pcg32 *generators, FloatVec *outputs, std::size_t num_generators) {
    const std::size_t global_thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_thread_index > num_generators) {
      return;
    }

    doapp::Pcg32 this_thread_generator = generators[global_thread_index];
    FloatVec &this_thread_output = outputs[global_thread_index];

    for (std::size_t i = 0; i < this_thread_output.size(); ++i) {
        this_thread_output[i] = this_thread_generator.rand01();
    }

    generators[global_thread_index] = this_thread_generator;
}

__host__ __device__ constexpr std::size_t divide_towards_positive_infinity(std::size_t numerator, std::size_t denominator) noexcept {
  return numerator / denominator + static_cast<std::size_t>(numerator % denominator != 0);
}

int main(int argc, const char *const argv[]) {
  if (argc < 2) {
    std::cerr << "error: missing required argument OUTPUT_FILE\n";

    return EXIT_FAILURE;
  }

  const char *const output_filename = argv[1];
  std::ofstream output_file(output_filename);

  if (!output_file.is_open()) {
    std::cerr << "error: couldn't open '" << output_filename << "' for writing\n";

    return EXIT_FAILURE;
  }

  static constexpr std::size_t NUM_GENERATORS = (1 << 22);
  static constexpr std::size_t NUMBERS_TO_GENERATE = (1 << 8);
  static constexpr std::size_t BLOCK_SIZE = 1024;
  static constexpr std::size_t NUM_BLOCKS = divide_towards_positive_infinity(NUM_GENERATORS, BLOCK_SIZE);
  static constexpr std::size_t TOTAL_NUMBERS = NUM_GENERATORS * NUMBERS_TO_GENERATE;

  doapp::Vector<doapp::Pcg32, doapp::Dynamic> generators =
      doapp::Pcg32::from_randomness(NUM_GENERATORS);

  FloatVecVec outputs(NUM_GENERATORS);

  for (std::size_t i = 0; i < NUM_GENERATORS; ++i) {
    outputs[i] = FloatVec(NUMBERS_TO_GENERATE);
  }

  auto start = std::chrono::steady_clock::now();
  generate_many_random<<<NUM_BLOCKS, BLOCK_SIZE>>>(generators.data(), outputs.data(), NUM_GENERATORS);
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();

  std::chrono::duration<double> elapsed = end - start;
  std::cout << "generated " << TOTAL_NUMBERS << " random floats in " << elapsed.count() << " seconds\n";

  start = std::chrono::steady_clock::now();
  const std::size_t num_threads = std::thread::hardware_concurrency() ?: 1;
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  std::vector<std::string> lines(NUM_GENERATORS);
  const std::size_t lines_per_thread = divide_towards_positive_infinity(NUM_GENERATORS, num_threads);

  for (std::size_t thread_index = 0; thread_index < num_threads; ++thread_index) {
    threads.emplace_back([&lines, &outputs, thread_index, lines_per_thread] {
      for (std::size_t i = thread_index * lines_per_thread; i < (thread_index + 1) * lines_per_thread && i < outputs.size(); ++i) {
        const FloatVec &this_line_data = outputs[i];

        std::ostringstream line;

        if (this_line_data.size() > 0) {
          line << std::hexfloat << this_line_data[0];

          for (std::size_t j = 1; j < this_line_data.size(); ++j) {
            line << ", " << this_line_data[j];
          }
        }

        line << '\n';

        lines[i] = line.str();
      }
    });
  }

  for (std::thread &t : threads) {
    t.join();
  }

  for (std::size_t i = 0; i < lines.size(); ++i) {
    output_file << lines[i];
  }

  output_file.flush();
  output_file.close();

  if (!output_file) {
    std::cerr << "error: couldn't write to file '" << output_filename << "'\n";
  }

  end = std::chrono::steady_clock::now();
  elapsed = end - start;

  std::cout << "printed " << TOTAL_NUMBERS << " random floats in " << elapsed.count() << " seconds\n";
}
