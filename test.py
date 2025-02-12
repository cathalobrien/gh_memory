import torch

# Load the allocator
new_alloc = torch.cuda.memory.CUDAPluggableAllocator('/p/home/jusers/obrien2/jedi/gh_memory/alloc.so', 'numa_malloc', 'my_free')
#'alloc.so', 'my_malloc', 'my_free')
# Swap the current allocator
torch.cuda.memory.change_current_allocator(new_alloc)

tensor = torch.randn(2,4, device="cuda", dtype=torch.float16) #16 Bytes


big_tensor = torch.randn(2,4,1000,1000,1000, device="cuda", dtype=torch.float16) #16GBytes

big_tensor[:] = 1
print(big_tensor)
tensor[:] = 1
print(tensor)
