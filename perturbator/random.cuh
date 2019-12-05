#ifndef RANDOM_CUH
#define RANDOM_CUH

#include "vector.cuh"
#include "common.cuh"

#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstring>
#include <system_error>
#include <vector>

#include <sys/random.h>

namespace doapp {

class Pcg32 {
public:
  Pcg32() noexcept = default;

  Pcg32(std::uint64_t initial_state, std::uint64_t initial_sequence) noexcept
      : state_(0), increment_((initial_sequence << 1) | 1) {
    step();
    state_ += initial_state;
    step();
  }

  static Vector<Pcg32, Dynamic> from_randomness(std::size_t n) {
    Vector<Pcg32, Dynamic> result(n);

    std::vector<std::uint64_t> initializers(n * 2);

    std::uint8_t *write_head =
        reinterpret_cast<std::uint8_t *>(initializers.data());
    std::uint8_t *const last =
        write_head + initializers.size() * sizeof(std::uint64_t);

    while (write_head < last) {
      const ssize_t this_num_initialized_or_err =
          getrandom(write_head, static_cast<std::size_t>(last - write_head), 0);

      if (this_num_initialized_or_err == -1) {
        throw std::system_error(errno, std::generic_category(),
                                "doapp::Pcg32::from_randomness: getrandom");
      }

      write_head += this_num_initialized_or_err;
    }

    for (std::size_t i = 0; i < n; ++i) {
      result[i] = Pcg32(initializers[2 * i], initializers[2 * i + 1]);
    }

    return result;
  }

  __host__ __device__ std::uint32_t rand() noexcept {
    const std::uint64_t old_state = state_;
    step();

    return output(old_state);
  }

  __host__ __device__ float rand01() noexcept {
    static constexpr std::size_t FRACTION_BITS = 23;
    static constexpr std::size_t EXPONENT_BIAS = 127;
    static constexpr std::size_t FLOAT_SIZE = sizeof(float) * CHAR_BIT;
    static constexpr std::size_t PRECISION = FRACTION_BITS + 1;
    static constexpr float SCALE = 1.0f / static_cast<float>(1_u32 << PRECISION);

    const std::uint32_t output = rand();
    const std::uint32_t mantissa = output >> (FLOAT_SIZE - PRECISION);

    return SCALE * static_cast<float>(mantissa + 1);
  }

  __host__ __device__ float rand11() noexcept {
    static constexpr std::size_t FRACTION_BITS = 23;
    static constexpr std::size_t EXPONENT_BIAS = 127;
    static constexpr std::size_t FLOAT_SIZE = sizeof(float) * CHAR_BIT;
    static constexpr std::size_t PRECISION = FRACTION_BITS + 1;
    static constexpr float SCALE = 1.0f / static_cast<float>(1_u32 << PRECISION);

    const std::uint32_t output = rand();
    const std::uint32_t mantissa = output >> (FLOAT_SIZE - PRECISION);
    const bool is_positive = (output & (1 << (FLOAT_SIZE - PRECISION - 1))) != 0;

    if (is_positive) {
      return SCALE * static_cast<float>(mantissa + 1);
    } else {
      return -SCALE * static_cast<float>(mantissa + 1);
    }

    /**
     *  @param min inclusive lower bound
     *  @param range if numbers are to be in [min, max), range is max - min
     */
    __host__ __device__ float rand_in_range(float min, float range) noexcept {
      assert(range > 0);

      return rand01() * range + min;
    }
  }

private:
  __host__ __device__ void step() noexcept {
    state_ *= 6364136223846793005ULL;
    state_ += increment_;
  }

  __host__ __device__ static std::uint32_t
  output(std::uint64_t state) noexcept {
    return rotr(((state >> 18) ^ state) >> 27, state >> 59);
  }

  __host__ __device__ static std::uint32_t rotr(std::uint32_t value,
                                                unsigned rot) noexcept {
    return (value >> rot) | (value << ((-rot) & 31));
  }

  std::uint64_t state_ = 0x853c49e6748fea9b;
  std::uint64_t increment_ = 0xda3e39cb94b95bdb;
};

} // namespace doapp

#endif
