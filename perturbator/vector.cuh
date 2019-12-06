#ifndef VECTOR_CUH
#define VECTOR_CUH

#include <cassert>
#include <cstddef>
#include <limits>
#include <new>

namespace doapp {

template <typename T> class Slice {
public:
  Slice() noexcept = default;

  __host__ __device__ Slice(T *data, std::size_t len) noexcept
      : data_(data), len_(len) {}

  __host__ __device__ Slice(T *data, std::size_t len,
                            std::size_t pitch) noexcept
      : data_(data), len_(len), pitch_(pitch) {}

  __host__ __device__ T &operator[](std::size_t i) const noexcept {
    assert(i < len_);

    return data_[pitch_ * i];
  }

  __host__ __device__ std::size_t size() const noexcept { return len_; }

  __host__ __device__ std::size_t pitch() const noexcept { return pitch_; }

  __host__ __device__ T *data() const noexcept { return data_; }

private:
  T *data_ = nullptr;
  std::size_t len_ = 0;
  std::size_t pitch_ = 1;
};

template <typename T, std::size_t N> class Vector {
public:
  Vector() noexcept = default;

  explicit Vector(std::size_t n) noexcept {
    assert(n == N);
  }

  __host__ __device__ T &operator[](std::size_t i) noexcept {
    assert(i < N);

    return data_[i];
  }

  __host__ __device__ const T &operator[](std::size_t i) const noexcept {
    assert(i < N);

    return data_[i];
  }

  __host__ __device__ std::size_t size() const noexcept { return N; }

  __host__ __device__ T *data() noexcept { return data_; }

  __host__ __device__ const T *data() const noexcept { return data_; }

  __host__ __device__ Slice<T> as_slice() noexcept { return {data_, N}; }

  __host__ __device__ Slice<const T> as_slice() const noexcept {
    return {data_, N};
  }

private:
  T data_[N] = {};
};

constexpr std::size_t Dynamic = std::numeric_limits<std::size_t>::max();

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

  __host__ __device__ T *data() noexcept { return data_; }

  __host__ __device__ const T *data() const noexcept { return data_; }

  __host__ __device__ Slice<T> as_slice() noexcept { return {data_, len_}; }

  __host__ __device__ Slice<const T> as_slice() const noexcept {
    return {data_, len_};
  }

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
