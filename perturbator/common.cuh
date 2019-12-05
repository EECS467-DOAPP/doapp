#ifndef COMMON_CUH
#define COMMON_CUH

#include <cstddef>
#include <cstdint>

namespace doapp {

using Scalar = float;

inline namespace literals {

__host__ __device__ constexpr std::uint8_t operator""_u8(unsigned long long x) noexcept {
    return static_cast<std::uint8_t>(x);
}

__host__ __device__ constexpr std::uint16_t operator""_u16(unsigned long long x) noexcept {
    return static_cast<std::uint16_t>(x);
}

__host__ __device__ constexpr std::uint32_t operator""_u32(unsigned long long x) noexcept {
    return static_cast<std::uint32_t>(x);
}

__host__ __device__ constexpr std::uint64_t operator""_u64(unsigned long long x) noexcept {
    return static_cast<std::uint64_t>(x);
}

__host__ __device__ constexpr std::int8_t operator""_i8(unsigned long long x) noexcept {
    return static_cast<std::int8_t>(x);
}

__host__ __device__ constexpr std::int16_t operator""_i16(unsigned long long x) noexcept {
    return static_cast<std::int16_t>(x);
}

__host__ __device__ constexpr std::int32_t operator""_i32(unsigned long long x) noexcept {
    return static_cast<std::int32_t>(x);
}

__host__ __device__ constexpr std::int64_t operator""_i64(unsigned long long x) noexcept {
    return static_cast<std::int64_t>(x);
}

} // inline namespace literals
} // namespace doapp

#endif
