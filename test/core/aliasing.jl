using Reactant, Test

function buffer_equals(x, y)
    if x isa Reactant.ConcreteIFRTArray
        y.data.buffer == x.data.buffer
    elseif x isa Reactant.ConcretePJRTArray
        all(((x, y),) -> x == y, zip(x.data, y.data))
    else
        error("invalid array type $(typeof(x))")
    end
end

function copy_with_broadcast!(a, b)
    a .= b
    return nothing
end

mutable struct X{T}
    x::T
    y::T
end

function br_func!(x, z)
    x.x .= z
    x.y = x.x
    return nothing
end

function f!(x, j)
    x.x .= 2.0 .* x.x
    x.y = x.x
    j .= x.y
    return nothing
end

@testset "Buffer aliasing" begin
    x = Reactant.to_rarray(ones(10))
    y = similar(x)
    @jit copy_with_broadcast!(y, x)
    @test !buffer_equals(x, y)

    x = Reactant.to_rarray(ones(10))
    y = Reactant.to_rarray(ones(10))
    x = X(x, y)
    z = Reactant.to_rarray(ones(10))
    @jit br_func!(x, z)
    @test buffer_equals(x.x, x.y)
    @test !buffer_equals(x.x, z)

    x = Reactant.to_rarray(ones(10))
    y = Reactant.to_rarray(ones(10))
    z = Reactant.to_rarray(ones(10))
    x = X(x, y)
    @jit f!(x, z)
    @test buffer_equals(x.x, x.y)
    @test !buffer_equals(x.y, z)
end

mutable struct State{A}
    u::A
    u⁰::A
end

function step!(state)
    state.u⁰ .= state.u
    state.u = state.u .+ 1.0
    return nothing
end

@testset "Mutable struct aliasing" begin
    sz = 3

    state_v = State(fill(0.5, sz), fill(0.0, sz))
    step!(state_v)

    state_r = State(ConcreteRArray(fill(0.5, sz)), ConcreteRArray(fill(0.0, sz)))

    r_step! = Reactant.@compile step!(state_r)
    r_step!(state_r)

    @test state_v.u == Array(state_r.u)
    @test state_v.u⁰ == Array(state_r.u⁰)
end

# https://github.com/EnzymeAD/Reactant.jl/issues/3469
# `copyto!(dest, src)` between concrete arrays writes back `dest.data = copy(src.data)`. That
# device copy is asynchronous; if the resulting buffer is not paired with its ready future,
# `wait(dest)` is a no-op and the raw-pointer scalar indexing path races with the in-flight
# copy (lost writes, reads of uninitialized memory).
@testset "copyto! result is synchronized before scalar access" begin
    x = Reactant.to_rarray(zeros(Float32, 560, 33, 2))
    v = view(x, :, 9:16, :)
    src = Reactant.to_rarray(ones(Float32, 560, 33, 2))

    for _ in 1:10
        copyto!(x, src)
        if x isa Reactant.ConcretePJRTArray
            # the copy must be tracked by a future, so that `wait`/`synchronize` cover it
            for buf in x.data
                @test buf.future !== nothing
            end
        end
        # `buffer_on_cpu` is only defined for PJRT buffers
        if x isa Reactant.ConcretePJRTArray && Reactant.buffer_on_cpu(x)
            Reactant.@allowscalar for I in CartesianIndices(v)
                v[I] *= 3
            end
            @test all(==(3), Array(v))
            @test all(==(1), Array(view(x, :, 1:8, :)))
        end
    end
end

function double_inplace_3023!(x)
    x .= x .* 2.0
    return x
end

function add_one_inplace_3023!(x)
    x .= x .+ 1.0
    return nothing
end

identity_3023(x) = x
tuple_both_3023!(x) = (x .= x .* 3.0; (x, x))

function save_old_then_mutate_3023!(s::State)
    old_u = s.u
    s.u = s.u .* 2.0
    return old_u
end

function mutate_field_and_return_3023!(s::State)
    s.u .= s.u .* 5.0
    return s.u
end

function push_and_return_3023!(x)
    push!(x, 4.0)
    return x
end

@testset "Returned array preserves identity (===) with argument (#3023)" begin
    x_ra = Reactant.to_rarray(ones(4))
    y = @jit double_inplace_3023!(x_ra)
    @test x_ra === y
    @test Array(x_ra) == [2.0, 2.0, 2.0, 2.0]
    @test Array(y) == [2.0, 2.0, 2.0, 2.0]

    @jit add_one_inplace_3023!(x_ra)
    @test x_ra === y
    @test Array(x_ra) == [3.0, 3.0, 3.0, 3.0]
    @test Array(y) == [3.0, 3.0, 3.0, 3.0]

    z = @jit identity_3023(x_ra)
    @test z === x_ra

    a, b = @jit tuple_both_3023!(x_ra)
    @test a === x_ra
    @test b === x_ra
    @test Array(a) == [9.0, 9.0, 9.0, 9.0]

    s = State(Reactant.to_rarray([1.0, 2.0]), Reactant.to_rarray([0.0, 0.0]))
    old_u = @jit save_old_then_mutate_3023!(s)
    @test Array(old_u) == [1.0, 2.0]
    @test Array(s.u) == [2.0, 4.0]
    @test old_u !== s.u

    u_ret = @jit mutate_field_and_return_3023!(s)
    @test u_ret === s.u
    @test Array(u_ret) == [10.0, 20.0]

    v_ra = Reactant.to_rarray([1.0, 2.0, 3.0])
    v_ret = @jit push_and_return_3023!(v_ra)
    @test v_ret === v_ra
    @test size(v_ra) == (4,)
    @test Array(v_ret) == [1.0, 2.0, 3.0, 4.0]
    @test Array(v_ra) == [1.0, 2.0, 3.0, 4.0]
end

# An array shared by many immutable wrappers, as when every field of a model holds its own
# copy of an immutable grid that references the same metric arrays.
struct SharedWrapper{A}
    data::A
end

struct ManyWrappers{W,A}
    wrappers::W
    other::A
end

function scale_other!(m)
    m.other .*= 2
    return nothing
end

function scale_shared!(m)
    first(m.wrappers).data .*= 2
    return nothing
end

@testset "Write-back to an array reachable through many paths" begin
    shared = Reactant.to_rarray(ones(4))
    other = Reactant.to_rarray(ones(4))
    m = ManyWrappers(ntuple(_ -> SharedWrapper(shared), 64), other)

    compiled = @compile scale_other!(m)
    body = string(only(code_lowered(compiled, Tuple{typeof(m)})))
    # the shared array is written back once, not once per path to it
    @test count("traced_setfield!", body) < 8

    compiled(m)
    @test all(==(2), Array(m.other))
    @test all(==(1), Array(shared))
    @test all(w -> w.data === shared, m.wrappers)

    @jit scale_shared!(m)
    @test all(==(2), Array(shared))
    @test all(w -> w.data === shared, m.wrappers)
    @test all(w -> all(==(2), Array(w.data)), m.wrappers)
end
