import triton
import triton.language as tl


# XXX: enable and support autotuning
# @triton.autotune(
#     configs=[
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": 128,
#                 "BLOCK_SIZE_N": 256,
#                 "BLOCK_SIZE_K": 64,
#                 "GROUP_SIZE_M": 8,
#             },
#             num_stages=3,
#             num_warps=8,
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": 64,
#                 "BLOCK_SIZE_N": 256,
#                 "BLOCK_SIZE_K": 32,
#                 "GROUP_SIZE_M": 8,
#             },
#             num_stages=4,
#             num_warps=4,
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": 128,
#                 "BLOCK_SIZE_N": 128,
#                 "BLOCK_SIZE_K": 32,
#                 "GROUP_SIZE_M": 8,
#             },
#             num_stages=4,
#             num_warps=4,
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": 128,
#                 "BLOCK_SIZE_N": 64,
#                 "BLOCK_SIZE_K": 32,
#                 "GROUP_SIZE_M": 8,
#             },
#             num_stages=4,
#             num_warps=4,
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": 64,
#                 "BLOCK_SIZE_N": 128,
#                 "BLOCK_SIZE_K": 32,
#                 "GROUP_SIZE_M": 8,
#             },
#             num_stages=4,
#             num_warps=4,
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": 128,
#                 "BLOCK_SIZE_N": 32,
#                 "BLOCK_SIZE_K": 32,
#                 "GROUP_SIZE_M": 8,
#             },
#             num_stages=4,
#             num_warps=4,
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": 64,
#                 "BLOCK_SIZE_N": 32,
#                 "BLOCK_SIZE_K": 32,
#                 "GROUP_SIZE_M": 8,
#             },
#             num_stages=5,
#             num_warps=2,
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": 32,
#                 "BLOCK_SIZE_N": 64,
#                 "BLOCK_SIZE_K": 32,
#                 "GROUP_SIZE_M": 8,
#             },
#             num_stages=5,
#             num_warps=2,
#         ),
#         # Good config for fp8 inputs.
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": 128,
#                 "BLOCK_SIZE_N": 256,
#                 "BLOCK_SIZE_K": 128,
#                 "GROUP_SIZE_M": 8,
#             },
#             num_stages=3,
#             num_warps=8,
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": 256,
#                 "BLOCK_SIZE_N": 128,
#                 "BLOCK_SIZE_K": 128,
#                 "GROUP_SIZE_M": 8,
#             },
#             num_stages=3,
#             num_warps=8,
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": 256,
#                 "BLOCK_SIZE_N": 64,
#                 "BLOCK_SIZE_K": 128,
#                 "GROUP_SIZE_M": 8,
#             },
#             num_stages=4,
#             num_warps=4,
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": 64,
#                 "BLOCK_SIZE_N": 256,
#                 "BLOCK_SIZE_K": 128,
#                 "GROUP_SIZE_M": 8,
#             },
#             num_stages=4,
#             num_warps=4,
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": 128,
#                 "BLOCK_SIZE_N": 128,
#                 "BLOCK_SIZE_K": 128,
#                 "GROUP_SIZE_M": 8,
#             },
#             num_stages=4,
#             num_warps=4,
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": 128,
#                 "BLOCK_SIZE_N": 64,
#                 "BLOCK_SIZE_K": 64,
#                 "GROUP_SIZE_M": 8,
#             },
#             num_stages=4,
#             num_warps=4,
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": 64,
#                 "BLOCK_SIZE_N": 128,
#                 "BLOCK_SIZE_K": 64,
#                 "GROUP_SIZE_M": 8,
#             },
#             num_stages=4,
#             num_warps=4,
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": 128,
#                 "BLOCK_SIZE_N": 32,
#                 "BLOCK_SIZE_K": 64,
#                 "GROUP_SIZE_M": 8,
#             },
#             num_stages=4,
#             num_warps=4,
#         ),
#     ],
#     key=["M", "N", "K"],
# )
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -----------------------------------------------------------
    # Add some integer bound assumptions.
    # This helps to guide integer analysis in the backend to optimize
    # load/store offset address calculation
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
