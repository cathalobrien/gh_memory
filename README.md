# gh_memory
# compile
```bash
module load CUDA
g++ alloc.cc -o alloc.so -I/usr/local/cuda/include -shared -fPIC
```

# using with torch

```python
# Load the allocator
new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
    'alloc.so', 'my_man_malloc', 'my_free')
    #'alloc.so', 'my_malloc', 'my_free')

# Swap the current allocator
torch.cuda.memory.change_current_allocator(new_alloc)
```
