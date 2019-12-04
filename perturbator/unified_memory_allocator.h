#ifndef UNIFIED_MEMORY_ALLOCATOR_H
#define UNIFIED_MEMORY_ALLOCATOR_H

#include <cstddef>
#include <new>
#include <vector>

#include <cuda.h>

namespace doapp {

template <typename T> class UnifiedMemoryAllocator {
public:
  static_assert(alignof(T) <= alignof(std::max_align_t));

  using value_type = T;

  T *allocate(std::size_t n) {
    void *ptr;
    const cudaError_t result = cudaMallocManaged(&ptr, sizeof(T) * n);

    if (result != cudaSuccess) {
      throw std::bad_alloc();
    }

    return static_cast<T *>(ptr);
  }

  void deallocate(T *ptr, std::size_t) { cudaFree(ptr); }
};

template <typename T>
using UnifiedMemoryVector = std::vector<T, UnifiedMemoryAllocator<T>>;

} // namespace doapp

#endif
