using Reactant, Test

using Reactant.Compiler: ThunkPlan, ThunkProgram, ResultWrite, AliasRead, AliasWrite

function with_runtime_thunk(f, enabled=true)
    previous = Reactant.Compiler.RUNTIME_THUNK[]
    Reactant.Compiler.RUNTIME_THUNK[] = enabled
    try
        return f()
    finally
        Reactant.Compiler.RUNTIME_THUNK[] = previous
    end
end

struct RTField{A,G}
    data::A
    grid::G
end

struct RTModel{F,C}
    fields::F
    clock::C
end

mutable struct RTPair{T}
    x::T
    y::T
end

function rt_model(n)
    grid = (; x=Reactant.to_rarray(collect(1.0:4.0)))
    names = ntuple(i -> Symbol(:f, i), n)
    fields = NamedTuple{names}(
        ntuple(i -> RTField(Reactant.to_rarray(fill(Float64(i), 4)), grid), n)
    )
    return RTModel(fields, Reactant.to_rarray([0.5]))
end

function rt_step!(model)
    total = zero(eltype(model.clock))
    for (i, field) in enumerate(model.fields)
        if isodd(i)
            field.data .= field.data .* 2 .+ field.grid.x .* model.clock
        else
            total += sum(field.data)
        end
    end
    model.clock .+= 1
    return total
end

function rt_alias!(pair, z)
    pair.x .= 2.0 .* z
    pair.y = pair.x
    return nothing
end

function rt_fill!(x, y)
    x .= 2
    return nothing
end

rt_sum(x, ys) = x .+ ys[1] .* ys[2]

rt_arrays(model) = (map(f -> Array(f.data), values(model.fields)), Array(model.clock))

@testset "Runtime thunk" begin
    @testset "Shared paths and in-place updates" begin
        results = map((false, true)) do enabled
            with_runtime_thunk(enabled) do
                model = rt_model(6)
                step! = @compile rt_step!(model)
                # the thunk is called with other arguments than the ones it was compiled with
                other = rt_model(6)
                totals = (step!(other), step!(other))
                (
                    map(t -> Reactant.to_number(t), totals),
                    rt_arrays(other),
                    rt_arrays(model),
                )
            end
        end
        @test results[1] == results[2]
        fields, clock = results[2][2]
        @test clock == [2.5]
        @test fields[2] == fill(2.0, 4)
        @test fields[1] == (fill(1.0, 4) .* 2 .+ (1.0:4.0) .* 0.5) .* 2 .+ (1.0:4.0) .* 1.5
    end

    @testset "Aliased arguments" begin
        with_runtime_thunk() do
            pair = RTPair(Reactant.to_rarray(ones(3)), Reactant.to_rarray(zeros(3)))
            z = Reactant.to_rarray(fill(3.0, 3))
            @jit rt_alias!(pair, z)
            @test Array(pair.x) == fill(6.0, 3)
            @test Array(pair.y) == fill(6.0, 3)
            @test pair.x.data === pair.y.data
            @test Array(z) == fill(3.0, 3)
        end
    end

    @testset "Donated and preserved arguments" begin
        with_runtime_thunk() do
            a = Reactant.to_rarray(ones(2, 2))
            b = Reactant.to_rarray(3 * ones(2, 2))
            fill! = @compile rt_fill!(a, b)
            fill!(a, b)
            fill!(a, b)
            @test Array(a) == 2 * ones(2, 2)
            @test Array(b) == 3 * ones(2, 2)

            x = Reactant.to_rarray([1.0, 2.0])
            ys = (Reactant.to_rarray([3.0, 4.0]), Reactant.to_rarray([5.0, 6.0]))
            @test Array(@jit(rt_sum(x, ys))) == [16.0, 26.0]
            @test Array(x) == [1.0, 2.0]
        end
    end

    @testset "Program" begin
        function plan()
            p = ThunkPlan(Tuple[(1, :fields, :a, :data), (1, :fields, :b, :data), (2,)], 3)
            push!(p.result_writes, ResultWrite(1, (1, :fields, :a, :data)))
            push!(p.result_writes, ResultWrite(2, (2,)))
            push!(p.alias_reads, AliasRead((1, :fields, :a, :data), false, false))
            push!(p.alias_writes, AliasWrite(1, (1, :fields, :b, :data)))
            return p
        end
        @test plan() == plan()
        @test hash(plan()) == hash(plan())
        program = Reactant.Compiler.intern_plan(plan())
        @test program === Reactant.Compiler.intern_plan(plan())
        @test program isa ThunkProgram
        # args[1], .fields, .a, .data, .b, .data and args[2]
        @test length(program.nodes) == 7
        @test all(i -> program.nodes[i].parent < i, eachindex(program.nodes))
        @test program.input_paths[1] == (:args, 1, :fields, :a, :data)
        # the write-back follows the last two steps of a path again
        @test program.nodes[program.result_writes[1].target.node].field == :fields
        @test program.result_writes[1].path == (:fields, :a, :data)
        @test program.result_writes[2].path == ()
        @test program.alias_reads[1].source.fields == (:a, :data)
        @test program.alias_writes[1].target.fields == (:b, :data)
    end
end
