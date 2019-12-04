#ifndef VECTOR_H
#define VECTOR_H

#include <cassert>
#include <cstddef>
#include <new>

#include <cuda.h>

namespace doapp {

template <typename T, std::ptrdiff_t N> class Vector {
public:
  __host__ __device__ T &operator[](std::size_t i) noexcept {
    assert(i < N);

    return data_[i];
  }

  __host__ __device__ const T &operator[](std::size_t i) const noexcept {
    assert(i < N);

    return data_[i];
  }

  __host__ __device__ std::size_t size() const noexcept {
    return static_cast<std::size_t>(N);
  }

private:
  T data_[N] = {};
};

constexpr std::ptrdiff_t Dynamic = -1;

template <typename T> class Vector<T, Dynamic> {
public:
  Vector() noexcept = default;

  explicit Vector(std::size_t n) {
    if (n == 0) {
      return;
    }

    void *ptr;

    if (cudaMallocManaged(&ptr, n * sizeof(T)) != cudaSuccess) {
      throw std::bad_alloc();
    }

    data_ = static_cast<T *>(ptr);
    len_ = n;

    for (std::size_t i = 0; i < len_; ++i) {
      try {
        ::new (&data_[i]) T();
      } catch (...) {
        for (std::size_t j = i - 1; j < i; --j) {
          data_[i].T::~T();
        }

        throw;
      }
    }
  }

  Vector(Vector &&other) noexcept : data_(other.data_), len_(other.len_) {
    other.data_ = nullptr;
    other.len_ = 0;
  }

  ~Vector() { clear(); }

  Vector &operator=(Vector &&other) noexcept {
    clear();

    data_ = other.data_;
    len_ = other.len_;

    other.data_ = nullptr;
    other.len_ = 0;

    return *this;
  }

  __host__ __device__ T &operator[](std::size_t i) noexcept {
    assert(i < len_);

    return data_[i];
  }

  __host__ __device__ const T &operator[](std::size_t i) const noexcept {
    assert(i < len_);

    return data_[i];
  }

  __host__ __device__ std::size_t size() const noexcept { return len_; }

  void clear() {
    if (data_ && len_ > 0) {
      for (std::size_t i = 0; i < len_; ++i) {
        data_[i].T::~T();
      }

      cudaFree(data_);
    }

    data_ = nullptr;
    len_ = 0;
  }

private:
  T *data_ = nullptr;
  std::size_t len_ = 0;
};

} // namespace doapp

#endif
