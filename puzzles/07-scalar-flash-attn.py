"""
Puzzle 07: Scalar FlashAttention
==============
From softmax to FlashAttention, we just need some computation.

Category: ["official"]
Difficulty: ["medium"]
"""

import tilelang
import tilelang.language as T
import torch

from common.utils import bench_puzzle, test_puzzle

"""
Now we have conquered softmax / online softmax, we can now implement one of the most important
operator in LLMs: FlashAttention.

To ensure a progressive learning experience, we will implement a scalar version of FlashAttention.
And we also remove the multi-head attention part. So in total we only have two dimensions: batch
size B and sequence length S, which are aligned with N, M in the previous puzzle. After such
simplification, you will find we are not so far from the FlashAttention algorithm. And with
TileLang, we can easily extend it to the full FlashAttention.

06-1: Simplified Scalar Flash Attention.

Inputs:
    Q: Tensor([B, S], float32)  # input tensor
    K: Tensor([B, S], float32)  # input tensor
    V: Tensor([B, S], float32)  # input tensor
    B: int   # batch size dimension. 1 <= B <= 256
    S: int   # sequence length dimension. 1 <= S <= 16384

Output:
    O: Tensor([B, S], float32)  # output tensor

Intermediates:
    MAX: float32  # max value of each row
    SUM: float32  # summation of each row
    QK: Tensor([B, S], float32)  # results of q*k
    P:  Tensor([B, S], float32)  # results of softmax(q*k) (not divided by summation).

Definition:
    for i in range(B):
        SUM = 0
        MAX = -inf
        for j in range(S):
            QK[i, j] = Q[i, j] * K[i, j]
            MAX = max(QK[i, j], MAX)
        for j in range(S):
            P[i, j] = exp(QK[i, j] - MAX)
            SUM += P[i, j]
        for j in range(M):
            O[i, j] = P[i, j] / SUM * V[i, j]
"""


def ref_scalar_flash_attn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    assert len(Q.shape) == 2
    assert len(K.shape) == 2
    assert len(V.shape) == 2
    assert Q.shape[0] == K.shape[0] == V.shape[0]  # B
    assert Q.shape[1] == K.shape[1] == V.shape[1]  # S
    assert Q.dtype == K.dtype == V.dtype == torch.float32
    return torch.softmax(Q * K, dim=1).mul_(V)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def tl_scalar_flash_attn(Q, K, V, BLOCK_B: int, BLOCK_S: int):
    log2_e = 1.44269504
    B, S = T.const("B, S")
    dtype = T.float32
    Q: T.Tensor((B, S), dtype)
    K: T.Tensor((B, S), dtype)
    V: T.Tensor((B, S), dtype)
    O = T.empty((B, S), dtype)

    with T.Kernel(T.ceildiv(B, BLOCK_B), threads=256) as bx:
        bidx = bx * BLOCK_B 

        r_max = T.alloc_fragment((BLOCK_B,), dtype)
        r_sum = T.alloc_fragment((BLOCK_B,), dtype)
        T.fill(r_max, float('-inf'))
        T.fill(r_sum, 0)

        r_q = T.alloc_fragment((BLOCK_B, BLOCK_S), dtype)
        r_k = T.alloc_fragment((BLOCK_B, BLOCK_S), dtype)
        r_v = T.alloc_fragment((BLOCK_B, BLOCK_S), dtype)
        r_tmp = T.alloc_fragment((BLOCK_B, BLOCK_S), dtype)

        # 第一轮循环，计算QK和MAX，r_tmp记录QK
        for s in T.Serial(T.ceildiv(S, BLOCK_S)):
            T.copy(Q[bidx, s*BLOCK_S], r_q)
            T.copy(K[bidx, s*BLOCK_S], r_k)
            for i, j in T.Parallel(BLOCK_B, BLOCK_S):
                r_tmp[i, j] = r_q[i, j] * r_k[i, j]
            T.reduce_max(r_tmp, r_max, dim=1, clear=False)   
            T.copy(r_tmp, O[bidx, s*BLOCK_S]) 


        # 第二轮循环，计算P和SUM，r_tmp记录P
        for s in T.Serial(T.ceildiv(S, BLOCK_S)):
            T.copy(O[bidx, s*BLOCK_S], r_tmp)
            for i, j in T.Parallel(BLOCK_B, BLOCK_S):
                r_tmp[i, j] = T.exp(r_tmp[i, j] - r_max[i])
            T.reduce_sum(r_tmp, r_sum, dim=1, clear=False)
            T.copy(r_tmp, O[bidx, s*BLOCK_S])

        # 第三轮循环
        for s in T.Serial(T.ceildiv(S, BLOCK_S)):
            T.copy(O[bidx, s*BLOCK_S], r_tmp)
            T.copy(V[bidx, s*BLOCK_S], r_v)
            for i, j in T.Parallel(BLOCK_B, BLOCK_S):
                r_tmp[i, j] = r_tmp[i, j] / r_sum[i] * r_v[i, j]
            T.copy(r_tmp, O[bidx, s*BLOCK_S])

    return O


def run_scalar_flash_attn():
    print("\n=== Scalar Flash Attention ===\n")
    B = 256
    S = 16384
    BLOCK_B = 16
    BLOCK_S = 128
    test_puzzle(
        tl_scalar_flash_attn,
        ref_scalar_flash_attn,
        {"B": B, "S": S, "BLOCK_B": BLOCK_B, "BLOCK_S": BLOCK_S},
    )
    bench_puzzle(
        tl_scalar_flash_attn,
        ref_scalar_flash_attn,
        {"B": B, "S": S, "BLOCK_B": BLOCK_B, "BLOCK_S": BLOCK_S},
        bench_torch=True,
    )


if __name__ == "__main__":
    run_scalar_flash_attn()
