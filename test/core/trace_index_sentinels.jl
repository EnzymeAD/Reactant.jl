using Reactant, Test

function trace_end_branch(x)
    @trace if sum(x) > 0
        y = x[2:end]
    else
        y = x[2:end] .* 0
    end
    return y
end

function trace_begin_branch(x)
    @trace if sum(x) > 0
        y = x[begin:3]
    else
        y = x[begin:3] .* 0
    end
    return y
end

function trace_index_offsets(x)
    @trace if sum(x) > 0
        y = x[(begin + 1):(end - 1)]
    else
        y = x[(begin + 1):(end - 1)] .* 0
    end
    return y
end

function trace_end_condition(x)
    @trace if sum(x[2:end]) > 0
        y = x[2:4]
    else
        y = x[2:4] .* 0
    end
    return y
end

function trace_begin_condition(x)
    @trace if sum(x[begin:3]) > 0
        y = x[2:4]
    else
        y = x[2:4] .* 0
    end
    return y
end

function trace_condition_and_branch(x)
    @trace if sum(x[(begin + 1):end]) > 0
        y = x[(begin + 1):end]
    else
        y = x[(begin + 1):end] .* 0
    end
    return y
end

function trace_matrix_indices(x)
    @trace if sum(x[begin:end, 2:end]) > 0
        y = x[(begin + 1):(end - 1), (begin + 1):end]
    else
        y = x[(begin + 1):(end - 1), (begin + 1):end] .* 0
    end
    return y
end

function trace_nested_indices(x)
    @trace if sum(x) > 0
        if sum(x[2:end]) > 2
            y = x[2:end]
        else
            y = -x[2:end]
        end
    else
        y = x[2:end] .* 0
    end
    return y
end

function trace_index_assignment(x)
    @trace y = if sum(x) > 0
        x[2:end]
    else
        x[2:end] .* 0
    end
    return y
end

@testset "@trace indexing begin/end" begin
    cases = (
        (trace_end_branch, x -> sum(x) > 0 ? x[2:end] : x[2:end] .* 0),
        (trace_begin_branch, x -> sum(x) > 0 ? x[begin:3] : x[begin:3] .* 0),
        (trace_index_offsets, x -> sum(x) > 0 ? x[(begin + 1):(end - 1)] : x[(begin + 1):(end - 1)] .* 0),
        (trace_end_condition, x -> sum(x[2:end]) > 0 ? x[2:4] : x[2:4] .* 0),
        (trace_begin_condition, x -> sum(x[begin:3]) > 0 ? x[2:4] : x[2:4] .* 0),
        (trace_condition_and_branch, x -> sum(x[(begin + 1):end]) > 0 ? x[(begin + 1):end] : x[(begin + 1):end] .* 0),
        (trace_index_assignment, x -> sum(x) > 0 ? x[2:end] : x[2:end] .* 0),
    )
    for T in (Float32, Float64)
        pos = T[0.2, 0.4, 0.6, 0.8]
        for (f, expected) in cases
            @testset "$(nameof(f)) $T" begin
                rx = Reactant.to_rarray(pos)
                exe = @compile f(rx)
                # Reuse the executable with both sides of the live predicate.
                for x in (pos, -pos)
                    reference = expected(x)
                    @test f(x) == reference
                    output = Array(exe(Reactant.to_rarray(x)))
                    @test output == reference
                    @test size(output) == size(reference)
                    @test eltype(output) === T
                end
            end
        end
        @testset "multidimensional $T" begin
            pos_matrix = reshape(T.(1:20), 4, 5)
            exe = @compile trace_matrix_indices(Reactant.to_rarray(pos_matrix))
            for x in (pos_matrix, -pos_matrix)
                reference = sum(x[begin:end, 2:end]) > 0 ?
                    x[(begin + 1):(end - 1), (begin + 1):end] :
                    x[(begin + 1):(end - 1), (begin + 1):end] .* 0
                @test trace_matrix_indices(x) == reference
                output = Array(exe(Reactant.to_rarray(x)))
                @test output == reference
                @test size(output) == (2, 4)
                @test eltype(output) === T
            end
        end
        @testset "nested branches $T" begin
            exe = @compile trace_nested_indices(Reactant.to_rarray(pos))
            for x in (pos, pos .* 2, -pos)
                reference = sum(x) > 0 ? (sum(x[2:end]) > 2 ? x[2:end] : -x[2:end]) : x[2:end] .* 0
                @test trace_nested_indices(x) == reference
                output = Array(exe(Reactant.to_rarray(x)))
                @test output == reference
                @test eltype(output) === T
            end
        end
    end
end
