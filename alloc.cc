// filename alloc.cc
#include <cuda_runtime_api.h>
#include <iostream>
#include <cstdint>
#include <numa.h>

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
	//cudaMallocAsync(&ptr, size, stream);
        cudaMalloc(&ptr, size);
        //std::cout<<"alloc "<<ptr<<" size: "<<size<<" bytes"<<std::endl;
   }
   return ptr;
}

//runs with CUDA_LAUNCH_BLOCKING=1
//othwise, errors with :
//[rank0]:     return forward_call(*args, **kwargs)
//[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//[rank0]:   File "/p/home/jusers/obrien2/jedi/raps/build/sources/anemoi-core/models/src/anemoi/models/layers/block.py", line 564, in forward
//[rank0]:     out = torch.cat([self.projection(chunk) for chunk in torch.tensor_split(out + x_r, num_chunks, dim=0)], dim=0)
//[rank0]:                                                                             ~~~~^~~~~
//[rank0]: RuntimeError: CUDA error: an illegal memory access was encountered


//n320 performance is .79 without and 0.09 with (malloc) :(    (both launched with 'CUDA_LAUNCH_BLOCKING=1'
//0.18349 with numa_alloc_onnode (4)
//0.04 o1280 128 1g numa_alloc_onnode 4, 0.168 normal
// numa_alloc_onnode 0 just fails silently with no error
void* numa_malloc(ssize_t size, int device, cudaStream_t stream) {
   void *ptr;
   //example: https://github.com/luigifusco/gh_benchmark/blob/github/alloc.cpp
   const std::uint64_t FOUR_GB = 4ULL * 1024 * 1024 * 1024; // 4 GB as a 64-bit unsigned literal
   if (size > FOUR_GB) { //if larger then 4GB allocate with managed memory
           ptr = numa_alloc_onnode( (size_t) size, 4); //Will have a have a unique so for each GPU
           //ptr = malloc( (size_t) size);
   }
   else {
	   cudaMallocAsync(&ptr, size, stream);
           //cudaMalloc(&ptr, size);
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
   const std::uint64_t FOUR_GB = 4ULL * 1024 * 1024 * 1024; // 4 GB as a 64-bit unsigned literal
   if (size > FOUR_GB) {
	   //free(ptr);
	   numa_free(ptr, (size_t) size);
   }
   else {
           cudaFree(ptr);
   }
}

}
