#ifndef MATRIX_CUH
#define MATRIX_CUH

#include "vector.cuh"

#include <cstddef>

namespace doapp {

template <typename T, std::size_t N, std::size_t M> class Matrix {
public:
  Matrix() noexcept = default;

  Matrix(std::size_t num_rows, std::size_t num_cols) noexcept
      : data_(num_rows * num_cols) {
    assert(num_rows == N);
    assert(num_cols == M);
  }

  __host__ __device__ Slice<T> operator[](std::size_t i) noexcept {
    assert(i < N);

    return {data_.data() + i * M, M};
  }

  __host__ __device__ Slice<const T> operator[](std::size_t i) const noexcept {
    assert(i < N);

    return {data_.data() + i * M, M};
  }

  __host__ __device__ Slice<T> col(std::size_t j) noexcept {
    assert(j < M);

    return {data_.data() + j, N, M};
  }

  __host__ __device__ Slice<const T> col(std::size_t j) const noexcept {
    assert(j < M);

    return {data_.data() + j, N, M};
  }

  __host__ __device__ std::size_t size() const noexcept { return N * M; }

  __host__ __device__ std::size_t num_rows() const noexcept { return N; }

  __host__ __device__ std::size_t num_cols() const noexcept { return M; }

  __host__ __device__ T *data() noexcept { return data_.data(); }

  __host__ __device__ const T *data() const noexcept { return data_.data(); }

  __host__ __device__ Slice<T> as_slice() noexcept { return data_.as_slice(); }

  __host__ __device__ Slice<const T> as_slice() const noexcept {
    return data_.as_slice();
  }

private:
  Vector<T, N * M> data_;
};

template <typename T> class Matrix<T, Dynamic, Dynamic> {
public:
  Matrix() noexcept = default;

  Matrix(std::size_t num_rows, std::size_t num_cols)
      : data_(num_rows * num_cols), num_rows_(num_rows), num_cols_(num_cols) {}

  __host__ __device__ Slice<T> operator[](std::size_t i) noexcept {
    assert(i < num_rows_);

    return {data_.data() + i * num_cols_, num_cols_};
  }

  __host__ __device__ Slice<const T> operator[](std::size_t i) const noexcept {
    assert(i < num_rows_);

    return {data_.data() + i * num_cols_, num_cols_};
  }

  __host__ __device__ Slice<T> col(std::size_t j) noexcept {
    assert(j < num_cols_);

    return {data_.data() + j, num_rows_, num_cols_};
  }

  __host__ __device__ Slice<const T> col(std::size_t j) const noexcept {
    assert(j < num_cols_);

    return {data_.data() + j, num_rows_, num_cols_};
  }

  __host__ __device__ std::size_t size() const noexcept {
    return num_rows_ * num_cols_;
  }

  __host__ __device__ std::size_t num_rows() const noexcept {
    return num_rows_;
  }

  __host__ __device__ std::size_t num_cols() const noexcept {
    return num_cols_;
  }

  __host__ __device__ T *data() noexcept { return data_.data(); }

  __host__ __device__ const T *data() const noexcept { return data_.data(); }

  __host__ __device__ Slice<T> as_slice() noexcept { return data_.as_slice(); }

  __host__ __device__ Slice<const T> as_slice() const noexcept {
    return data_.as_slice();
  }

private:
  Vector<T, Dynamic> data_;
  std::size_t num_rows_ = 0;
  std::size_t num_cols_ = 0;
};

template <typename T, std::size_t N> class Matrix<T, N, Dynamic> {
public:
  Matrix() noexcept = default;

  Matrix(std::size_t num_rows, std::size_t num_cols)
      : data_(N * num_cols), num_cols_(num_cols) {
    assert(num_rows == N);
  }

  __host__ __device__ Slice<T> operator[](std::size_t i) noexcept {
    assert(i < N);

    return {data_.data() + i * num_cols_, num_cols_};
  }

  __host__ __device__ Slice<const T> operator[](std::size_t i) const noexcept {
    assert(i < N);

    return {data_.data() + i * num_cols_, num_cols_};
  }

  __host__ __device__ Slice<T> col(std::size_t j) noexcept {
    assert(j < num_cols_);

    return {data_.data() + j, N, num_cols_};
  }

  __host__ __device__ Slice<const T> col(std::size_t j) const noexcept {
    assert(j < num_cols_);

    return {data_.data() + j, N, num_cols_};
  }

  __host__ __device__ std::size_t size() const noexcept {
    return N * num_cols_;
  }

  __host__ __device__ std::size_t num_rows() const noexcept { return N; }

  __host__ __device__ std::size_t num_cols() const noexcept {
    return num_cols_;
  }

  __host__ __device__ T *data() noexcept { return data_.data(); }

  __host__ __device__ const T *data() const noexcept { return data_.data(); }

  __host__ __device__ Slice<T> as_slice() noexcept { return data_.as_slice(); }

  __host__ __device__ Slice<const T> as_slice() const noexcept {
    return data_.as_slice();
  }

private:
  Vector<T, Dynamic> data_;
  std::size_t num_cols_ = 0;
};

template <typename T, std::size_t M> class Matrix<T, Dynamic, M> {
public:
  Matrix() noexcept = default;

  Matrix(std::size_t num_rows, std::size_t num_cols)
      : data_(num_rows * M), num_rows_(num_rows) {
    assert(num_cols == M);
  }

  __host__ __device__ Slice<T> operator[](std::size_t i) noexcept {
    assert(i < num_rows_);

    return {data_.data() + i * M, M};
  }

  __host__ __device__ Slice<const T> operator[](std::size_t i) const noexcept {
    assert(i < num_rows_);

    return {data_.data() + i * M, M};
  }

  __host__ __device__ Slice<T> col(std::size_t j) noexcept {
    assert(j < M);

    return {data_.data() + j, num_rows_, M};
  }

  __host__ __device__ Slice<const T> col(std::size_t j) const noexcept {
    assert(j < M);

    return {data_.data() + j, num_rows_, M};
  }

  __host__ __device__ std::size_t size() const noexcept {
    return num_rows_ * M;
  }

  __host__ __device__ std::size_t num_rows() const noexcept {
    return num_rows_;
  }

  __host__ __device__ std::size_t num_cols() const noexcept { return M; }

  __host__ __device__ T *data() noexcept { return data_.data(); }

  __host__ __device__ const T *data() const noexcept { return data_.data(); }

  __host__ __device__ Slice<T> as_slice() noexcept { return data_.as_slice(); }

  __host__ __device__ Slice<const T> as_slice() const noexcept {
    return data_.as_slice();
  }

private:
  Vector<T, Dynamic> data_;
  std::size_t num_rows_ = 0;
};

} // namespace doapp

#endif
