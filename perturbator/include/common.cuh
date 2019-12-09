#ifndef COMMON_CUH
#define COMMON_CUH

#include <cstddef>
#include <cstdint>

namespace doapp {
//joint angle size declarations
//TODO: since this has to be declared at compile time, consider making dimensionality of waypoints/noise vectors also known at compile time
constexpr int num_joints = 5;
extern __constant__ float min_joint_angles[num_joints];
extern __constant__ float max_joint_angles[num_joints];
extern float host_min_joint_angles[num_joints];
extern float host_max_joint_angles[num_joints];

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
