import triton
import triton.language as tl


@triton.jit
def seeded_dropout_kernel(
    x_ptr,
    output_ptr,
    mask_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # randomly prune it
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # write-back
    output = tl.where(x_keep, x / (1 - p), 0.0)
    mask_out = tl.where(x_keep, 1.0, 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)
    tl.store(mask_ptr + offsets, mask_out, mask=mask)
