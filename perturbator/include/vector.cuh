#ifndef VECTOR_CUH
#define VECTOR_CUH

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>

namespace doapp {
namespace detail {

struct UninitializedTag {};

} // namespace detail

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

  __host__ __device__ void swap(Slice<T> &other) {
    assert(len_ == other.len_);

    using std::swap;

    swap(data_, other.data_);
    swap(len_, other.len_);
    swap(pitch_, other.pitch_);
  }

private:
  T *data_ = nullptr;
  std::size_t len_ = 0;
  std::size_t pitch_ = 1;
};

template <typename T>
__host__ __device__ void swap(Slice<T> &lhs, Slice<T> &rhs) {
  lhs.swap(rhs);
}

template <typename T, std::size_t N> class Vector {
public:
  __host__ __device__ Vector(std::initializer_list<T> list) noexcept(
      std::is_nothrow_copy_constructible<T>::value)
      : Vector(detail::UninitializedTag{}) {
    assert(list.size() == N);

    for (auto iter = list.begin(); len_ < N; ++len_, ++iter) {
      ::new (data() + len_) T(*iter);
    }
  }

  __host__ __device__ explicit Vector(std::size_t n) noexcept(
      std::is_nothrow_default_constructible<T>::value)
      : Vector(detail::UninitializedTag{}) {
    assert(n == N);

    for (; len_ < N; ++len_) {
      ::new (data() + len_) T();
    }
  }

  __host__ __device__ explicit Vector(detail::UninitializedTag) noexcept {}

  __host__ __device__ ~Vector() {
    for (std::size_t i = 0; i < len_; ++i) {
      data()[i].T::~T();
    }
  }

  __host__ __device__ T &operator[](std::size_t i) noexcept {
    assert(i < N);

    return data()[i];
  }

  __host__ __device__ const T &operator[](std::size_t i) const noexcept {
    assert(i < N);

    return data()[i];
  }

  __host__ __device__ std::size_t size() const noexcept { return N; }

  __host__ __device__ T *data() noexcept {
    return reinterpret_cast<T *>(&data_);
  }

  __host__ __device__ const T *data() const noexcept {
    return reinterpret_cast<const T *>(&data_);
  }

  __host__ __device__ Slice<T> as_slice() noexcept { return {data(), N}; }

  __host__ __device__ Slice<const T> as_slice() const noexcept {
    return {data(), N};
  }

private:
  typename std::aligned_storage<sizeof(T) * N, alignof(T)>::type data_;
  std::size_t len_ = 0;
};

constexpr std::size_t Dynamic = 0;

template <typename T> class Vector<T, Dynamic> {
public:
  Vector() noexcept = default;

  __host__ __device__ Vector(std::initializer_list<T> list) noexcept(
      std::is_nothrow_copy_constructible<T>::value)
      : Vector(detail::UninitializedTag{}, list.size()) {
    for (auto iter = list.begin(); iter < list.end(); ++len_, ++iter) {
      ::new (data() + len_) T(*iter);
    }
  }

  explicit Vector(std::size_t n) : Vector(detail::UninitializedTag{}, n) {
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

  Vector(detail::UninitializedTag, std::size_t n) {
    if (n == 0) {
      return;
    }

    void *ptr;

    if (cudaMallocManaged(&ptr, n * sizeof(T)) != cudaSuccess) {
      throw std::bad_alloc();
    }

    data_ = static_cast<T *>(ptr);
    capacity_ = n;
  }

  ~Vector() {
    clear();

    if (data_) {
      cudaFree(data_);
    }
  }

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

  void clear() noexcept {
    if (data_) {
      assert(capacity_ > 0);

      for (std::size_t i = 0; i < len_; ++i) {
        data_[i].T::~T();
      }
    }

    len_ = 0;
  }

private:
  T *data_ = nullptr;
  std::size_t len_ = 0;
  std::size_t capacity_ = 0;
};

template <typename T, std::size_t N>
__host__ __device__ Vector<T, N>
operator+(const Vector<T, N> &v,
          const Vector<T, N> &u) noexcept(noexcept(Vector<T, N>(0))) {
  assert(v.size() == u.size());

  Vector<T, N> w(v.size());

  for (std::size_t i = 0; i < v.size(); ++i) {
    w[i] = v[i] + u[i];
  }

  return w;
}

template <typename T, std::size_t N>
__host__ __device__ Vector<T, N>
operator-(const Vector<T, N> &v,
          const Vector<T, N> &u) noexcept(noexcept(Vector<T, N>(0))) {
  assert(v.size() == u.size());

  Vector<T, N> w(v.size());

  for (std::size_t i = 0; i < v.size(); ++i) {
    w[i] = v[i] - u[i];
  }

  return w;
}

template <typename T, std::size_t N>
__host__ __device__ T dot(const Vector<T, N> &v,
                          const Vector<T, N> &u) noexcept(noexcept(T())) {
  assert(v.size() == u.size());

  T result();

  for (std::size_t i = 0; i < v.size(); ++i) {
    result += v[i] * u[i];
  }

  return result;
}

} // namespace doapp

#endif
