module load CUDA
g++ alloc.cc -o alloc.so -I/usr/local/cuda/include -shared -fPIC
