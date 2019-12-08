#ifndef UNIQUE_PTR_CUH
#define UNIQUE_PTR_CUH

#include <cassert>
#include <new>

namespace doapp {

template <typename T>
class UniquePtr {
public:
    UniquePtr() noexcept = default;

    explicit UniquePtr(T *ptr) noexcept : ptr_(ptr) { }

    explicit UniquePtr(UniquePtr<T> &&other) noexcept : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }

    ~UniquePtr() {
        if (!ptr_) {
            return;
        }

        ptr_->T::~T();
        cudaFree(ptr_);
    }

    UniquePtr<T> &operator=(UniquePtr<T> &&other) noexcept {
        if (ptr_) {
            ptr_->T::~T();
            cudaFree(ptr_);
        }

        ptr_ = other.ptr_;
        other.ptr_ = nullptr;

        return *this;
    }

    __host__ __device__ T *get() const noexcept {
        return ptr_;
    }

    __host__ __device__ T &operator*() const noexcept {
        assert(ptr_);

        return *ptr_;
    }

    __host__ __device__ T *operator->() const noexcept {
        assert(ptr_);

        return ptr_;
    }

    __host__ __device__ explicit operator bool() const noexcept {
        return ptr_ != nullptr;
    }

    __host__ __device__ void swap(UniquePtr<T> &other) noexcept {
        T *const temp = ptr_;
        ptr_ = other.ptr_;
        other.ptr_ = temp;
    }

private:
    T *ptr_ = nullptr;
};

template <typename T>
__host__ __device__ void swap(UniquePtr<T> &lhs, UniquePtr<T> &rhs) noexcept {
    lhs.swap(rhs);
}

template <typename T, typename ...Ts>
UniquePtr<T> make_unique(Ts &&...ts) {
    void *ptr;

    if (cudaMallocManaged(&ptr, sizeof(T)) != cudaSuccess) {
        throw std::bad_alloc();
    }

    try {
        ::new(ptr) T(std::forward<Ts>(ts)...);
    } catch (...) {
        cudaFree(ptr);

        throw;
    }

    return UniquePtr<T>(static_cast<T *>(ptr));
}

} // namespace doapp

#endif
