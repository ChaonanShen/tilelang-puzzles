"""
Puzzle 01: Copy
==============
This puzzle asks you to implement a copy operation that copies data from one
tensor to another.

Category: ["official"]
Difficulty: ["easy"]
"""

import tilelang
import tilelang.language as T
import torch

from common.utils import bench_puzzle, test_puzzle

"""
To begin with, we start to provide a runnable example of TileLang's copy.
The code below shows how to define a 1-D copy kernel using TileLang. We assume
all tensors are stored in the global memory (DRAM) of GPU initially.

01-1: 1-D copy kernel.

Inputs:
    A: Tensor([N,], float16)  # input tensor
    N: int   # size of the tensor. 1 <= N <= 1024*1024

Output:
    B: Tensor([N,], float16)  # copied tensor

Definition:
    for i in range(N):
        B[i] = A[i]
"""


def ref_copy_1d(A: torch.Tensor):
    assert len(A.shape) == 1
    assert A.dtype == torch.float16
    return A.clone()


"""
We will use TileLang's EagerJIT kernel programming style to provide a better programming experience.

In TileLang, a kernel is defined as a Python function decorated with @tilelang.jit. This decorator
enables JIT compilation of the kernel. The function parameters represent the input/output tensors
(fully compacted torch Tensors) and other hyperparameters of the kernel. In this example, the input
tensor A is passed as a parameter, and the output tensor B is returned as the result.

After the function declaration, the host code section defines constants and tensor shapes/dtypes.

Next, we need to specify the kernel launch configuration. In TileLang, we use T.Kernel to launch
a kernel. It accepts a list of blocks indicating the number of blocks to launch, and an integer
threads specifying the number of threads per block. The kernel function will be launched with a
total of blocks*threads threads.

In this first step, we will write a simple serial copy kernel that launches only one thread.
"""


@tilelang.jit
def tl_copy_1d_serial(A):
    # The host/declaration part of TileLang script.
    N = T.const("N")
    A: T.Tensor((N,), T.float16)
    B = T.empty((N,), T.float16)

    # The body of the kernel function is written in TileLang DSL.
    # We use T.Kernel to launch a kernel.
    with T.Kernel(1, threads=1) as _:
        # Here T.copy is a built-in TileOp in TileLang.
        # It will automatically utilize available threads in the block
        # to do efficient memory copy (including auto parallelism and vectorization)
        # As we only launch one thread here, it will be lowered into a serial loop copy
        # with certain bit width vectorization (like 128 bits per copy).
        T.copy(A, B)

    return B


def run_copy_1d_serial(N: int):
    print("\n=== Copy 1D Serial ===\n")
    test_puzzle(tl_copy_1d_serial, ref_copy_1d, {"N": N})


"""
The implementation above only launches a single thread, which is not efficient.
Now we want to launch multiple threads within a single kernel to copy data in parallel.

Since T.copy automatically parallelizes copying inside a block, we don't need many
modifications to make it work.

Now, try changing the number of threads per block to 128 or 256 and compare the
speedup you achieve.
"""


@tilelang.jit
def tl_copy_1d_multi_threads(A):
    # The host/declaration part of TileLang script.
    N = T.const("N")
    A: T.Tensor((N,), T.float16)
    B = T.empty((N,), T.float16)

    with T.Kernel(1, threads=128) as _:
        T.copy(A, B)

    return B


def run_copy_1d_multi_threads(N: int):
    print("\n=== Copy 1D Multi-threads ===\n")

    test_puzzle(tl_copy_1d_multi_threads, ref_copy_1d, {"N": N})

    # This may take a while since N is large
    bench_puzzle(
        tl_copy_1d_serial,
        ref_copy_1d,
        {"N": N},
        bench_name="TL Serial",
        bench_torch=True,
    )
    bench_puzzle(
        tl_copy_1d_multi_threads,
        ref_copy_1d,
        {"N": N},
        bench_name="TL Multi-threads",
        bench_torch=False,
    )


"""
Finally, we want to parallelize the copy operation across multiple blocks.
We use BLOCK_N to represent the number of elements each block should copy.
The rest of the implementation is similar to the previous version. We assume that N is divisible
by BLOCK_N.

Note: You will need to handle the memory access ranges for different blocks. Fortunately,
we have `bx` (the block index) available, so you can compute the start and end indices for
each block accordingly.
"""


@tilelang.jit
def tl_copy_1d_parallel(A, BLOCK_N: int):
    # The host/declaration part of TileLang script.
    N = T.const("N")
    A: T.Tensor((N,), T.float16)
    B = T.empty((N,), T.float16)

    with T.Kernel(T.ceildiv(N, BLOCK_N), threads=128) as bx:
        T.copy(A[bx*BLOCK_N : (bx+1)*BLOCK_N], B[bx*BLOCK_N : (bx+1)*BLOCK_N])

    return B


def run_copy_1d_parallel(N: int):
    print("\n=== Copy 1D Parallel ===\n")
    BLOCK_N = 1024
    test_puzzle(tl_copy_1d_parallel, ref_copy_1d, {"N": N, "BLOCK_N": BLOCK_N})
    bench_puzzle(
        tl_copy_1d_parallel,
        ref_copy_1d,
        {"N": N, "BLOCK_N": BLOCK_N},
        bench_name="TL Parallel",
        bench_torch=True,
    )


if __name__ == "__main__":
    # serial 是单 block 单线程，没有 layout inference 限制，可以测极小值
    serial_sizes = [
        1,              # 极小，只有 serial 能跑
        1024,
        1000,
        1025,
        1024 * 256,
        1024 * 1024,
        3000,
    ]

    # multi_threads 单 block 128 线程，N 至少要够 128 线程分，上限也不能太大
    multi_thread_sizes = [
        1024,
        1000,
        1025,
        3000,
        1024 * 256,
    ]

    # parallel 多 block，能扩展到大 N，但同样不能测 N=1
    parallel_sizes = [
        1024,
        1000,
        1025,
        1024 * 256,
        1024 * 1024,
        3000,
    ]

    for N in serial_sizes:
        run_copy_1d_serial(N)

    for N in multi_thread_sizes:
        run_copy_1d_multi_threads(N)

    for N in parallel_sizes:
        run_copy_1d_parallel(N)
