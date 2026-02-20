using CUDA, Reactant
using cuTile: cuTile
const ct = cuTile

function vadd(a, b, c, tile_size::Int)
    pid = ct.bid(1)
    tile_a = ct.load(a, pid, (tile_size,))
    tile_b = ct.load(b, pid, (tile_size,))
    ct.store(c, pid, tile_a + tile_b)
    return nothing
end

vector_size = 2^12
tile_size = 16

a, b = CUDA.rand(Float32, vector_size), CUDA.rand(Float32, vector_size)
c = CUDA.zeros(Float32, vector_size)

ct.launch(vadd, (cld(vector_size, tile_size), 1, 1), a, b, c, ct.Constant(tile_size))
