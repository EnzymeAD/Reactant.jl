using Reactant
using OffsetArrays
using Test

struct GridSpacing1229{DX,DY}
    dx::DX
    dy::DY
end

struct KernelSpacingOp1229{G} <: AbstractArray{Float64,3}
    grid::G
    nz::Int
end

function Base.size(op::KernelSpacingOp1229)
    return (length(axes(op.grid.dx, 1)) - 2, length(axes(op.grid.dx, 2)) - 2, op.nz)
end

@inline function Base.getindex(op::KernelSpacingOp1229, i, j, k)
    return (op.grid.dx[i, j] + op.grid.dy[i, j]) * k
end

struct ConditionalOp1229{O,C} <: AbstractArray{Float64,3}
    operand::O
    cutoff::C
    mask_value::Float64
end

Base.size(c::ConditionalOp1229) = size(c.operand)

@inline function Base.getindex(c::ConditionalOp1229, i, j, k)
    val = c.operand[i, j, k]
    return ifelse(val > c.cutoff, val, c.mask_value)
end

# Subtypes `AbstractArray{Any,3}` just like `Oceananigans.AbstractOperations.ConditionalOperation`
struct AnyConditionalOp1229{O,C} <: AbstractArray{Any,3}
    operand::O
    cutoff::C
    mask_value::Float64
end

Base.size(c::AnyConditionalOp1229) = size(c.operand)

@inline function Base.getindex(c::AnyConditionalOp1229, i, j, k)
    val = c.operand[i, j, k]
    return ifelse(val > c.cutoff, val, c.mask_value)
end

struct ZeroDimLazyOp1229{A} <: AbstractArray{Any,0}
    arr::A
end

Base.size(::ZeroDimLazyOp1229) = ()
@inline Base.getindex(z::ZeroDimLazyOp1229) = z.arr[1] + z.arr[2]

function mapfirst_any_eltype_lazy_1229!(dest, dx, dy, cutoff)
    grid = GridSpacing1229(dx, dy)
    kop = KernelSpacingOp1229(grid, 3)
    cop = AnyConditionalOp1229(kop, cutoff, 999.0)
    Base.mapfirst!(identity, view(dest, :, :, 1:1), cop)
    return dest
end

function assign_subarray_from_lazy_1229!(dest, cond_op)
    vdest = view(dest, :, :, 1:1)
    vsrc = view(cond_op, :, :, 2:2)
    vdest .= vsrc
    return dest
end

function assign_dropped_dim_view_1229!(dest2d, cond_op)
    dest2d .= view(cond_op, 2:4, :, 2)
    return dest2d
end

function sum_reshaped_permuted_lazy_1229(dx, dy)
    cop = AnyConditionalOp1229(KernelSpacingOp1229(GridSpacing1229(dx, dy), 3), 20.0, 999.0)
    perm = PermutedDimsArray(cop, (3, 1, 2))
    flat = reshape(perm, length(cop))
    out = similar(parent(dx), Float64, length(cop))
    out .= flat
    return out
end

function read_zerodim_lazy_1229(a)
    z = ZeroDimLazyOp1229(a)
    out = similar(a, Float64, ())
    out .= z
    return out
end

@testset "Lazy AbstractArray holding traced leaves (#1229)" begin
    dx_host = OffsetArray(reshape(collect(1.0:30.0), 6, 5), 0:5, 0:4)
    dy_host = OffsetArray(fill(0.5, 6, 5), 0:5, 0:4)
    grid_host = GridSpacing1229(dx_host, dy_host)
    kop_host = KernelSpacingOp1229(grid_host, 3)
    cop_host = ConditionalOp1229(kop_host, 20.0, 999.0)
    any_cop_host = AnyConditionalOp1229(kop_host, 20.0, 999.0)

    dx_r = OffsetArray(Reactant.to_rarray(parent(dx_host)), 0:5, 0:4)
    dy_r = OffsetArray(Reactant.to_rarray(parent(dy_host)), 0:5, 0:4)
    grid_r = GridSpacing1229(dx_r, dy_r)
    kop_r = KernelSpacingOp1229(grid_r, 3)
    cop_r = ConditionalOp1229(kop_r, ConcreteRNumber(20.0), 999.0)

    # 1a. Eager `Base.mapfirst!` into a SubArray of ConcreteRArray
    dest_host = zeros(Float64, 4, 3, 2)
    Base.mapfirst!(identity, view(dest_host, :, :, 1:1), cop_host)

    dest_r = Reactant.to_rarray(zeros(Float64, 4, 3, 2))
    Base.mapfirst!(identity, view(dest_r, :, :, 1:1), cop_r)
    @test Array(dest_r) ≈ dest_host

    # 1b. `Base.mapfirst!` on `AbstractArray{Any,3}` SubArray (`eltype == Any`)
    dest_any_host = zeros(Float64, 4, 3, 2)
    Base.mapfirst!(identity, view(dest_any_host, :, :, 1:1), any_cop_host)

    dest_any_r = Reactant.to_rarray(zeros(Float64, 4, 3, 2))
    @jit mapfirst_any_eltype_lazy_1229!(dest_any_r, dx_r, dy_r, ConcreteRNumber(20.0))
    @test Array(dest_any_r) ≈ dest_any_host

    # 2. Eager `map!` with non-identity function on KernelSpacingOp
    dest_kop_host = zeros(Float64, 4, 3, 1)
    map!(abs2, view(dest_kop_host, :, :, 1:1), view(kop_host, :, :, 3:3))

    dest_kop_r = Reactant.to_rarray(zeros(Float64, 4, 3, 1))
    map!(abs2, view(dest_kop_r, :, :, 1:1), view(kop_r, :, :, 3:3))
    @test Array(dest_kop_r) ≈ dest_kop_host

    # 3. `@jit` broadcast assignment from SubArray of lazy ConditionalOp
    dest_jit_host = zeros(Float64, 4, 3, 2)
    assign_subarray_from_lazy_1229!(dest_jit_host, cop_host)

    dest_jit_r = Reactant.to_rarray(zeros(Float64, 4, 3, 2))
    @jit assign_subarray_from_lazy_1229!(dest_jit_r, cop_r)
    @test Array(dest_jit_r) ≈ dest_jit_host

    # 4. `@jit` view with both UnitRange and dropped scalar index
    dest2d_host = zeros(Float64, 3, 3)
    assign_dropped_dim_view_1229!(dest2d_host, cop_host)

    dest2d_r = Reactant.to_rarray(zeros(Float64, 3, 3))
    @jit assign_dropped_dim_view_1229!(dest2d_r, cop_r)
    @test Array(dest2d_r) ≈ dest2d_host

    # 5. PermutedDimsArray + ReshapedArray over `AbstractArray{Any,3}` lazy traced array
    flat_host = collect(
        Float64, reshape(PermutedDimsArray(any_cop_host, (3, 1, 2)), length(any_cop_host))
    )
    flat_r = @jit sum_reshaped_permuted_lazy_1229(dx_r, dy_r)
    @test Array(flat_r) ≈ flat_host

    # 6. 0-dimensional lazy traced array
    v_r = Reactant.to_rarray([3.5, 4.25])
    z_out = @jit read_zerodim_lazy_1229(v_r)
    @test Array(z_out)[] ≈ 7.75
end
