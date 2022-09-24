from numba import cuda
import numpy as np
import math

@cuda.jit
def increment_by_one(array):
    pos = cuda.grid(1)
    if pos < array.size:
        array[pos] += 1

my_array = np.random.random(10000)

print(f"My array mean: {my_array.mean}")

threads_per_block = 32

blocks_per_grid = (my_array.size + (threads_per_block - 1)) // threads_per_block
print(my_array.size, threads_per_block, blocks_per_grid)

increment_by_one[blocks_per_grid, threads_per_block](my_array)

print(f"My array +1 mean: {my_array.mean}")

# notes
# block size (number of threads per block) is often crucial
# from a software perspective, the number of threads per block determines how many threads are given an area of shared memory
# from a hardware perspective, the number of threads must be large enough for full occupation of execution units (see https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

