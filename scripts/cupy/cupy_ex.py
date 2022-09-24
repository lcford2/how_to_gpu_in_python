import cupy as cp
import numpy as np
from time import perf_counter as timer

# Ab = c (solve for b)

def array_creation(size):
    # array creation
    s = timer()
    x_cpu = np.random.random((size, size, size))
    e = timer()
    print(f"Array created of size: {x_cpu.nbytes / 1000 / 1000} MB")
    print(f"Array Creation CPU Time: {e-s:.4} seconds")

    s = timer()
    x_gpu = cp.random.random((size, size, size))
    # sync GPU and CPU to make sure data gen is done
    cp.cuda.Stream.null.synchronize()
    e = timer()
    print(f"Array Creation GPU Time: {e-s:.4} seconds")
    return x_cpu, x_gpu


def array_scalar_multiply(x_cpu, x_gpu, scalar):
    s = timer()
    x_cpu *= scalar
    e = timer()
    print(f"Array Multiplication CPU Time: {e-s:.4} seconds")

    s = timer()
    x_gpu *= scalar
    # sync GPU and CPU to make sure data gen is done
    cp.cuda.Stream.null.synchronize()
    e = timer()
    print(f"Array Multiplication GPU Time: {e-s:.4} seconds")


def multiple_operations(x_cpu, x_gpu):
    s = timer()
    x_cpu *= 1.5
    x_cpu += x_cpu
    x_cpu *= x_cpu
    mean = x_cpu.mean()
    e = timer()
    print(f"Multiple Ops CPU Time: {e-s:.4} seconds; Answer={mean:.2f}")

    s = timer()
    x_gpu *= 1.5
    x_gpu += x_gpu
    x_gpu *= x_gpu
    mean = x_gpu.mean()
    # sync GPU and CPU to make sure data gen is done
    cp.cuda.Stream.null.synchronize()
    e = timer()
    print(f"Multiple Ops GPU Time: {e-s:.4} seconds; Answer={mean:.2f}")


def solve_linear_system(size):
    X = np.random.random((size, size))
    y = np.random.random((size))
    s = timer()
    x = np.linalg.solve(X, y)
    e = timer()
    print(f"CPU Linear System Solve: {e - s:.4f} seconds; Answer Mean: {x.mean():.4f}")

    X = cp.random.random((size, size))
    y = cp.random.random((size))
    cp.cuda.Stream.null.synchronize()
    s = timer()
    x = cp.linalg.solve(X, y)
    cp.cuda.Stream.null.synchronize()
    e = timer()
    print(f"GPU Linear System Solve: {e - s:.4f} seconds; Answer Mean: {x.mean():.4f}")


def matrix_multiplication(size):
    X = np.random.random((size, size))
    s = timer()
    result = np.matmul(X, X)
    e = timer()
    print(f"CPU Mat Mul: {e - s:.4f} seconds; Answer Mean: {result.mean():.4f}")

    # X = cp.random.random((size, size))
    X = cp.asarray(X)
    cp.cuda.Stream.null.synchronize()
    s = timer()
    result = cp.empty_like(X)
    cp.matmul(X, X, out=result)
    cp.cuda.Stream.null.synchronize()
    e = timer()
    print(f"GPU Mat Mul: {e - s:.4f} seconds; Answer Mean: {result.mean():.4f}")


if __name__ == "__main__":
    x_cpu, x_gpu = array_creation(1000)
    print()
    array_scalar_multiply(x_cpu, x_gpu, 10)
    print()
    multiple_operations(x_cpu, x_gpu)
    print()
    solve_linear_system(1000)
    print()
    matrix_multiplication(1000)









