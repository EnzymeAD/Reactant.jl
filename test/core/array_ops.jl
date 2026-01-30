# Tests for array operations
using Reactant, Test

const RunningOnTPU = contains(string(Reactant.devices()[1]), "TPU")

@testset "concatenation" begin
    @testset "Number" begin
        x = fill(true)
        x_concrete = Reactant.to_rarray(x)

        test_vcat(x) = begin
            x = x[]
            [x; x; x]
        end
        y = @jit test_vcat(x_concrete)
        @test y == test_vcat(x)
        @test eltype(y) === Bool

        test_hcat(x) = begin
            x = x[]
            [x x x]
        end
        y = @jit test_hcat(x_concrete)
        @test y == test_hcat(x)
        @test eltype(y) === Bool

        test_hvcat(x) = begin
            x = x[]
            [x x x; x x x]
        end
        y = @jit test_hvcat(x_concrete)
        @test y == test_hvcat(x)
        @test eltype(y) === Bool

        test_hvncat(x) = begin
            x = x[]
            [x x x; x x x;;; x x x; x x x]
        end
        y = @jit test_hvncat(x_concrete)
        @test y == test_hvncat(x)
        @test eltype(y) === Bool

        test_typed_vcat(x) = begin
            x = x[]
            Int[x; x; x]
        end
        y = @jit test_typed_vcat(x_concrete)
        @test y == test_typed_vcat(x)
        @test eltype(y) === Int

        test_typed_hcat(x) = begin
            x = x[]
            Int[x x x]
        end
        y = @jit test_typed_hcat(x_concrete)
        @test y == test_typed_hcat(x)
        @test eltype(y) === Int

        test_typed_hvcat(x) = begin
            x = x[]
            Int[x x x; x x x]
        end
        y = @jit test_typed_hvcat(x_concrete)
        @test y == test_typed_hvcat(x)
        @test eltype(y) === Int

        test_typed_hvncat(x) = begin
            x = x[]
            Int[x x x; x x x;;; x x x; x x x]
        end
        y = @jit test_typed_hvncat(x_concrete)
        @test y == test_typed_hvncat(x)
        @test eltype(y) === Int
    end

    @testset "$(ndims(x))-dim Array" for x in [
        fill(true),
        [true, false],
        [true false],
        [true true; true false],
        [
            true true true true; true true true false;;;
            true true false true; true true false false;;;
            true false true true; true false true false
        ],
    ]
        x_concrete = Reactant.to_rarray(x)

        test_vcat(x) = [x; x; x]
        y = @jit test_vcat(x_concrete)
        @test y == test_vcat(x)
        @test eltype(y) === Bool

        test_hcat(x) = [x x x]
        y = @jit test_hcat(x_concrete)
        @test y == test_hcat(x)
        @test eltype(y) === Bool

        test_hvcat(x) = [x x x; x x x]
        y = @jit test_hvcat(x_concrete)
        @test y == test_hvcat(x)
        @test eltype(y) === Bool

        test_hvncat(x) = [x x x; x x x;;; x x x; x x x]
        y = @jit test_hvncat(x_concrete)
        @test y == test_hvncat(x)
        @test eltype(y) === Bool

        test_typed_vcat(x) = Int[x; x; x]
        y = @jit test_typed_vcat(x_concrete)
        @test y == test_typed_vcat(x)
        @test eltype(y) === Int

        test_typed_hcat(x) = Int[x x x]
        y = @jit test_typed_hcat(x_concrete)
        @test y == test_typed_hcat(x)
        @test eltype(y) === Int

        test_typed_hvcat(x) = Int[x x x; x x x]
        y = @jit test_typed_hvcat(x_concrete)
        @test y == test_typed_hvcat(x)
        @test eltype(y) === Int

        test_typed_hvncat(x) = Int[x x x; x x x;;; x x x; x x x]
        y = @jit test_typed_hvncat(x_concrete)
        @test y == test_typed_hvncat(x)
        @test eltype(y) === Int
    end

    @testset "Number and RArray" for a in [1.0f0, 1.0e0]
        typeof_a = typeof(a)
        _b = typeof_a.([2.0, 3.0, 4.0])
        _c = typeof_a.([2.0 3.0 4.0])
        b = Reactant.to_rarray(_b)
        c = Reactant.to_rarray(_c)

        y = @jit vcat(a, b)
        @test y == vcat(a, _b)
        @test y isa ConcreteRArray{typeof_a,1}

        y1 = @jit vcat(a, c')
        @test y1 == vcat(a, _c')
        @test y1 isa ConcreteRArray{typeof_a,2}

        z = @jit hcat(a, c)
        @test z == hcat(a, _c)
        @test z isa ConcreteRArray{typeof_a,2}

        z1 = @jit hcat(a, b')
        @test z1 == hcat(a, _b')
        @test z1 isa ConcreteRArray{typeof_a,2}
    end
end

@testset "repeat" begin
    @testset for (size, counts) in Iterators.product(
        [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)],
        [(), (1,), (2,), (2, 1), (1, 2), (2, 2), (2, 2, 2), (1, 1, 1, 1, 1)],
    )
        x = Reactant.TestUtils.construct_test_array(Float64, size...)

        @testset "outer repeat" begin
            @test (@jit repeat(Reactant.to_rarray(x), counts...)) ≈ repeat(x, counts...)
        end

        length(counts) < length(size) && continue

        @testset "inner repeat" begin
            @test (@jit repeat(Reactant.to_rarray(x); inner=counts)) ≈
                repeat(x; inner=counts)
        end
    end
end

@testset "repeat specialize" begin
    x_ra = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 2, 3))
    hlo = repr(@code_hlo(repeat(x_ra, 2, 3)))
    @test !contains(hlo, "stablehlo.dynamic_update_slice")
end

@testset "stack" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 4, 4)
    y = Reactant.TestUtils.construct_test_array(Float64, 4, 4)
    x_ra = Reactant.to_rarray(x)
    y_ra = Reactant.to_rarray(y)

    @test @jit(stack((x_ra, x_ra))) ≈ stack((x, x))
    @test @jit(stack((x_ra, x_ra); dims=2)) ≈ stack((x, x); dims=2)
    @test @jit(stack((x_ra, y_ra); dims=2)) ≈ stack((x, y); dims=2)
    @test @jit(stack((x_ra, y_ra, x_ra); dims=1)) ≈ stack((x, y, x); dims=1)

    @test @jit(stack((x, x))) isa Any
    @test @jit(stack((x, x); dims=2)) isa Any
    @test @jit(stack((x, y); dims=2)) isa Any
    @test @jit(stack((x, y, x); dims=1)) isa Any
end

@testset "unstable stack" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 4, 4)
    y = Reactant.TestUtils.construct_test_array(Float64, 4, 4)
    x_ra = Reactant.to_rarray(x)
    y_ra = Reactant.to_rarray(y)

    s1(x) = begin
        xs = []
        push!(xs, x)
        push!(xs, x)
        stack(xs)
    end
    s2(x) = begin
        xs = []
        push!(xs, x)
        push!(xs, x)
        stack(xs; dims=2)
    end
    s3(x, y) = begin
        xs = []
        push!(xs, x)
        push!(xs, y)
        stack(xs; dims=2)
    end
    s4(x, y) = begin
        xs = []
        push!(xs, x)
        push!(xs, y)
        push!(xs, x)
        stack(xs; dims=2)
    end

    @test @jit(s1(x_ra)) ≈ s1(x)
    @test @jit(s2(x_ra)) ≈ s2(x)
    @test @jit(s3(x_ra, y_ra)) ≈ s3(x, y)
    @test @jit(s4(x_ra, y_ra)) ≈ s4(x, y)

    @test @jit(s1(x)) isa Any
    @test @jit(s2(x)) isa Any
    @test @jit(s3(x, y)) isa Any
    @test @jit(s4(x, y)) isa Any
end

stack_numbers(x) = stack([sum(x[:, i]) for i in axes(x, 2)])

@testset "stack numbers" begin
    x = Reactant.TestUtils.construct_test_array(Float32, 2, 4)
    x_ra = Reactant.to_rarray(x)
    @test @jit(stack_numbers(x_ra)) ≈ stack_numbers(x)
end

@testset "collect" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 2, 3)
    x_ra = Reactant.to_rarray(x)

    @testset "Reactant.to_rarray" begin
        y = collect(x_ra)
        @test y ≈ x
        @test y !== x_ra
    end

    @testset "TracedRArray" begin
        y = @jit(collect(x_ra))
        @test y ≈ x
        @test y !== x_ra
    end

    x = 5
    x_ra = ConcreteRNumber(x)

    @testset "ConcreteRNumber" begin
        y = collect(x_ra)
        @test y isa Array{Int,0}
    end

    @testset "TracedRArray" begin
        y = @jit(collect(x_ra))
        @test y isa ConcreteRArray{Int,0}
        @test y == x
    end
end

similar_from_type(x) = similar(typeof(x), (4, 5))

@testset "similar" begin
    x = zeros(2, 3)
    y = Reactant.to_rarray(x)
    f = @compile similar(y)
    @test size(f(y)) == size(x)
    @test eltype(f(y)) == eltype(x)

    f_from_type = @compile similar_from_type(y)
    @test size(f_from_type(y)) == (4, 5)
    @test eltype(f_from_type(y)) == eltype(x)
end

@testset "similar Reactant.to_rarray" begin
    c = Reactant.to_rarray(ones(50, 70))
    sim_c = similar(c)
    @test typeof(sim_c) == typeof(c) && size(sim_c) == size(sim_c)
end

@testset "circshift" begin
    x = reshape(collect(Float32, 1:36), 2, 6, 3)
    x_ra = Reactant.to_rarray(x)

    @test @jit(circshift(x_ra, (1, 2))) ≈ circshift(x, (1, 2))
    @test @jit(circshift(x_ra, (1, 2, 3))) ≈ circshift(x, (1, 2, 3))
    @test @jit(circshift(x_ra, (-3, 2))) ≈ circshift(x, (-3, 2))
    @test @jit(circshift(x_ra, (5, 2))) ≈ circshift(x, (5, 2))
end

function meshgrid(args::AbstractVector...)
    return let N = length(args)
        stack(enumerate(args)) do (i, arg)
            new_shape = ones(Int, N)
            new_shape[i] = length(arg)
            repeat_sizes = collect(Int, map(length, args))
            repeat_sizes[i] = 1
            return repeat(reshape(arg, new_shape...), repeat_sizes...)
        end
    end
end

function meshgrid(x::Number, y::Number)
    return meshgrid(range(eltype(x)(0), x; length=10), range(eltype(y)(0), y; length=10))
end

@testset "meshgrid" begin
    x = 10.0f0
    y = 20.0f0
    x_ra = ConcreteRNumber(x)
    y_ra = ConcreteRNumber(y)

    @test @jit(meshgrid(x_ra, y_ra)) ≈ meshgrid(x, y)
end

@testset "copyto! ConcreteArray" begin
    x_ra = Reactant.to_rarray(ones(4, 4))
    y_ra = Reactant.to_rarray(zeros(2, 2))
    copyto!(view(x_ra, 1:2, 1:2), y_ra)
    @test Array(x_ra) ==
        [0.0 0.0 1.0 1.0; 0.0 0.0 1.0 1.0; 1.0 1.0 1.0 1.0; 1.0 1.0 1.0 1.0]
end

@testset "copyto! ConcreteArray Array" begin
    x_ra = Reactant.to_rarray(ones(4, 4))
    y_ra = view(zeros(4, 4), 1:2, 1:2)
    copyto!(view(x_ra, 1:2, 1:2), y_ra)
    @test Array(x_ra) ==
        [0.0 0.0 1.0 1.0; 0.0 0.0 1.0 1.0; 1.0 1.0 1.0 1.0; 1.0 1.0 1.0 1.0]
end

@testset "copyto! TracedRArray" begin
    x_ra = Reactant.to_rarray(ones(4, 4))
    y_ra = Reactant.to_rarray(zeros(2, 2))
    @jit copyto!(x_ra, 6, y_ra, 3, 2)

    x = ones(4, 4)
    y = zeros(2, 2)
    copyto!(x, 6, y, 3, 2)
    @test Array(x_ra) == x
end

reshapecopy!(x, y) = begin
    Base.copyto!(x, reshape(y, size(x)))
    nothing
end

@testset "copyto! Reshaped TracedRArray" begin
    x = zeros(3, 4, 5)
    y = collect(reshape(1:60, (3, 20)))

    xr = Reactant.to_rarray(x)
    yr = Reactant.to_rarray(y)

    @jit reshapecopy!(xr, yr)

    reshapecopy!(x, y)
    @test Array(xr) == x
end

@testset "copyto!, no offsets" begin
    a = Float32[10, 20, 30, 40, 50]
    len = length(a)
    b = Float32[111, 222, 333, 444, 555]
    cpu = fill(0.0f0, len)
    gpu = (Reactant.@jit Reactant.Ops.fill(0.0f0, (len,)))

    cpu .= a
    gpu .= b
    copyto!(cpu, gpu)
    @test gpu == b
    @test cpu == b

    cpu .= a
    gpu .= b
    copyto!(gpu, cpu)
    @test gpu == a
    @test cpu == a
end

@testset "copyto!, with offsets" begin
    a = Float32[10, 20, 30, 40, 50, 60, 70]
    alen = length(a)
    b = Float32[111, 222, 333, 444, 555]
    blen = length(b)

    dest = fill(0.0f0, alen)
    src = Reactant.@jit Reactant.Ops.fill(0.0f0, (blen,))

    for desto in 1:alen, srco in 1:blen, l in 1:min(blen - srco + 1, alen - desto + 1)
        if src isa ConcretePJRTArray
            expected = copyto!(copy(a), desto, b, srco, l)
            dest .= a
            src .= b
            copyto!(dest, desto, src, srco, l)
            @test dest == expected
        end
    end

    dest = Reactant.@jit Reactant.Ops.fill(0.0f0, (alen,))
    if dest isa ConcretePJRTArray
        src = fill(0.0f0, blen)
        for desto in 1:alen, srco in 1:blen, l in 1:min(blen - srco + 1, alen - desto + 1)
            expected = copyto!(copy(a), desto, b, srco, l)
            dest .= a
            src .= b
            copyto!(dest, desto, src, srco, l)
            @test dest == expected
        end
    end
end

@testset "copy(::Broadcast.Broadcasted{ArrayStyle{ConcreteRArray}})" begin
    x_ra = Reactant.to_rarray(ones(4, 4))
    res = copy(Broadcast.broadcasted(-, Broadcast.broadcasted(+, x_ra, 1)))
    @test res ≈ -(Array(x_ra) .+ 1)
end

@testset "copy/deepcopy" begin
    for op in (copy, deepcopy)
        x = Reactant.to_rarray(ones(4, 4))
        if x isa Reactant.ConcretePJRTArray
            orig_ptr = only(x.data).buffer.buffer
            y = op(x)
            @test y isa Reactant.ConcretePJRTArray
            @test only(y.data).buffer.buffer != orig_ptr
            @test only(x.data).buffer.buffer == orig_ptr
        else
            orig_ptr = x.data.buffer.buffer
            y = op(x)
            @test y isa Reactant.ConcreteIFRTArray
            @test y.data.buffer.buffer != orig_ptr
            @test x.data.buffer.buffer == orig_ptr
        end

        x = Reactant.to_rarray(4.0; track_numbers=Number)
        if x isa Reactant.ConcretePJRTNumber
            orig_ptr = only(x.data).buffer.buffer
            y = op(x)
            @test y isa Reactant.ConcretePJRTNumber
            @test only(y.data).buffer.buffer != orig_ptr
            @test only(x.data).buffer.buffer == orig_ptr
        else
            orig_ptr = x.data.buffer.buffer
            y = op(x)
            @test y isa Reactant.ConcreteIFRTNumber
            @test y.data.buffer.buffer != orig_ptr
            @test x.data.buffer.buffer == orig_ptr
        end
    end
end

@testset "map!" begin
    x = Reactant.TestUtils.construct_test_array(Float32, 2, 3)
    y = zeros(Float32, 2, 3)

    x_ra = Reactant.to_rarray(x)
    y_ra = Reactant.to_rarray(y)

    @test Array(@jit(map!(abs2, y_ra, x_ra))) ≈ map!(abs2, y, x)
    @test Array(y_ra) ≈ y
end

map_test_1(i, xᵢ, yᵢ) = xᵢ + yᵢ + max(xᵢ, yᵢ)

@testset "multi-argument map" begin
    x = collect(Float32, 1:10)
    y = collect(Float32, 31:40)

    x_ra = Reactant.to_rarray(x)
    y_ra = Reactant.to_rarray(y)

    gt = map(map_test_1, 1:length(x), x, y)
    @test @jit(map(map_test_1, 1:length(x), x_ra, y_ra)) ≈ gt

    z = similar(x)
    z_ra = Reactant.to_rarray(z)
    map!(map_test_1, z, 1:length(x), x, y)
    @jit map!(map_test_1, z_ra, 1:length(x), x_ra, y_ra)
    @test z ≈ z_ra
    @test z_ra ≈ gt
end

function f_row_major(x::AbstractArray{T}) where {T}
    y = [1 2; 3 4; 5 6]
    if x isa Reactant.TracedRArray
        y = Reactant.promote_to(Reactant.TracedRArray{Reactant.unwrapped_eltype(T),2}, y)
    end
    return x .+ y
end

@testset "array attributes: row major" begin
    x = zeros(Int, 3, 2)
    x_ra = Reactant.to_rarray(x)
    @test @jit(f_row_major(x_ra)) ≈ f_row_major(x)
end

@testset "duplicate args (#226)" begin
    first_arg(x, y) = x
    x_ra = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float64, 2, 2))
    res = @jit first_arg(x_ra, x_ra)
    @test res ≈ x_ra
end

@testset "traced size" begin
    x_ra = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float64, 5, 32, 7))
    @test @jit(size(x_ra, ConcreteRNumber(1))) == 5
    @test @jit(size(x_ra, ConcreteRNumber(2))) == 32
    @test @jit(size(x_ra, ConcreteRNumber(3))) == 7
end
