// filename alloc.cc
#include <cuda_runtime_api.h>
#include <iostream>
#include <cstdint> // For fixed-width integer types

extern "C" {


void* my_man_malloc(ssize_t size, int device, cudaStream_t stream) {
   void *ptr;
   const std::uint64_t FOUR_GB = 4ULL * 1024 * 1024 * 1024; // 4 GB as a 64-bit unsigned literal
   if (size > FOUR_GB) { //if larger then 4GB allocate with managed memory
         //The ETH paper uses numa_alloc_onnode from libnuma
        cudaMallocManaged(&ptr, size);
        std::cout<<"alloc (Managed) "<<ptr<<" size: "<<size<<" bytes"<<std::endl;

        //advice https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1ge37112fc1ac88d0f6bab7a945e48760a
        cudaMemAdvise(ptr, size, cudaMemAdviseSetReadMostly, device);
        cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, device);
        cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, device);

        cudaMemPrefetchAsync(ptr, size, device); //copy the data over to the gpu now, so it's there 'by default'
   }
   else {
        //the ETH paper uses cudaMallocAsync
        //https://arxiv.org/html/2408.11556v1#S4
        //cudaMallocAsync example: https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/streamOrderedAllocation/streamOrderedAllocation.cu
        cudaMallocAsync(&ptr, size, stream);
        //cudaMalloc(&ptr, size);
        //std::cout<<"alloc "<<ptr<<" size: "<<size<<" bytes"<<std::endl;
   }
   return ptr;
}

void* numa_malloc(ssize_t size, int device, cudaStream_t stream) {
   void *ptr;
   //example: https://github.com/luigifusco/gh_benchmark/blob/github/alloc.cpp
   const std::uint64_t FOUR_GB = 4ULL * 1024 * 1024 * 1024; // 4 GB as a 64-bit unsigned literal
   if (size > FOUR_GB) { //if larger then 4GB allocate with managed memory
           //ptr = numa_alloc_onnode(size, ) //would like to use this once i establish the numa nodes
           ptr = malloc(size);
   }
   else {
           cudaMallocAsync(&ptr, size, stream);
   }
   return ptr;

}

void* my_malloc(ssize_t size, int device, cudaStream_t stream) {
   void *ptr;
   cudaMalloc(&ptr, size);
   std::cout<<"alloc "<<ptr<<" size: "<<size<<" bytes"<<std::endl;
   return ptr;
}

void my_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
   //std::cout<<"free "<<ptr<< " "<<stream<<std::endl;
   cudaFree(ptr);
}

}
