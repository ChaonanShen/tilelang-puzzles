"""
Puzzle 05: Reduce Sum
==============
In this puzzle, you will learn how to do reduce in TileLang.

Category: ["official"]
Difficulty: ["easy"]
"""

import tilelang
import tilelang.language as T
import torch

from common.utils import bench_puzzle, test_puzzle

"""
We already do broadcasting in previous example. Now let's see how to do reduction. Luckily,
we don't need to implement detailed reduction logics since TileLang provides built-in
TileOps. Before this, T.copy is the only TileOp we have seen. But we have experienced that
with T.copy and T.Parallel we can already do many things!

HINT:
1. For reduction, we have `T.reduce` and `T.reduce_xxx`, where xxx represents the reduction
operation, e.g., `T.reduce_sum`. Note that for efficiency, we need to perform these TileOps
in the fragment buffers instead of global memory.
2. You may need a serial loop to do this puzzle. Use `T.Serial` to create a serial loop.
3. For numerical stability, we shift the data type to float32 for now.

05-1: Reduce sum.

Inputs:
    A: Tensor([N, M], float32)  # input tensor
    B: Tensor([N,], float32)  # input tensor
    N: int   # size of the tensor. 1 <= N <= 4096
    M: int   # size of the tensor. 1 <= M <= 16384

Output:
    B: Tensor([N,], float32)  # output tensor

Definition:
    for i in range(N):
        B[i] = 0
        for j in range(M):
            B[i] += A[i, j]
"""


def ref_reduce_sum(A: torch.Tensor):
    assert len(A.shape) == 2
    assert A.dtype == torch.float32
    return torch.sum(A, dim=1)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def tl_reduce_sum(A, BLOCK_N: int, BLOCK_M: int):
    N, M = T.const("N, M")
    dtype = T.float32
    A: T.Tensor((N, M), dtype)
    B = T.empty((N,), dtype)

    with T.Kernel(T.ceildiv(N, BLOCK_N), threads=256) as bx:
        nidx = bx * BLOCK_N

        r_a = T.alloc_fragment((BLOCK_N, BLOCK_M), dtype)
        r_b = T.alloc_fragment((BLOCK_N,), dtype)
        T.clear(r_b)

        # 一个block处理一个tile，每个tile逐块累加
        for k in T.Serial(T.ceildiv(M, BLOCK_M)):
            T.copy(A[nidx, k*BLOCK_M], r_a)
            T.reduce_sum(r_a, r_b, dim=1, clear=False)

        T.copy(r_b, B[nidx])

    return B


def run_reduce_sum():
    print("\n=== Reduce Sum ===\n")

    BLOCK_N = 16
    BLOCK_M = 128

    # 覆盖：正常大值、不整除 N、不整除 M、两者都不整除、边界小值
    test_cases = [
        (4096, 16384),   # 基准，完全整除
        (4000, 16384),   # N 不整除 BLOCK_N
        (4096, 16000),   # M 不整除 BLOCK_M
        (4000, 16000),   # N、M 都不整除
        (1,    1),       # 最小边界
        (1,    16384),   # N=1，M 最大
        (4096, 1),       # N 最大，M=1
        (16,   128),     # 刚好一个 block
        (17,   129),     # 比一个 block 多一点
        (100,  300),     # 中等大小，两者都不整除
    ]

    for N, M in test_cases:
        print(f"  Testing N={N}, M={M} ...")
        try:
            test_puzzle(
                tl_reduce_sum,
                ref_reduce_sum,
                {"N": N, "M": M, "BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M},
            )
            print(f"    PASSED")
        except Exception as e:
            print(f"    FAILED: {e}")

    # bench 只跑一次大 case
    print("\n--- Benchmark (N=4096, M=16384) ---")
    bench_puzzle(
        tl_reduce_sum,
        ref_reduce_sum,
        {"N": 4096, "M": 16384, "BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M},
        bench_torch=True,
    )


if __name__ == "__main__":
    run_reduce_sum()
