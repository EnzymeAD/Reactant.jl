using Reactant, Test

f1(x::AbstractMatrix) = sum(x; dims=1)

f2(x::AbstractMatrix, y::Int) = x .+ y

function f3(x::AbstractArray{T,3}, y::Int) where {T}
    return Reactant.Ops.batch(Base.Fix2(f2, y), x, [1, 2])
end

@testset "mapslices" begin
    A = collect(reshape(1:30, (2, 5, 3)))
    A_ra = Reactant.to_rarray(A)

    B = mapslices(f1, A; dims=[1, 2])
    B_ra = @jit mapslices(f1, A_ra; dims=[1, 2])

    @test B ≈ B_ra

    B = mapslices(sum, A; dims=[1, 3])
    B_ra = @jit mapslices(sum, A_ra; dims=[1, 3])

    @test B ≈ B_ra
end

@testset "closure" begin
    A = collect(reshape(1:30, (2, 5, 3)))
    A_ra = Reactant.to_rarray(A)

    @test @jit(f3(A_ra, 1)) ≈ A .+ 1
end

# Autobatching testing
struct TupleStruct{T1,T2}
    A::T1
    B::T2
end

struct NestedTupleStruct{T1,T2}
    C::T1
    foo::T2
end

function update(bar::NestedTupleStruct)
    A = bar.C * bar.foo.A
    B = bar.C * bar.foo.B
    return NestedTupleStruct(bar.C, TupleStruct(A, B))
end

@testset "Auto-Batching DotGeneral" begin
    A = rand(Float32, 32, 32, 4)
    B = rand(Float32, 32, 32, 4)
    C = rand(Float32, 32, 32, 4)
    structs = [
        NestedTupleStruct(
            @view(C[:, :, i]), TupleStruct(@view(A[:, :, i]), @view(B[:, :, i]))
        ) for i in 1:size(A, 3)
    ]
    structs_ra = Reactant.to_rarray(structs)

    hlo = repr(@code_hlo broadcast(update, structs_ra))
    @test count("dot_general", hlo) == 2
end
