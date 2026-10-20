if abspath(PROGRAM_FILE) == @__FILE__
    include("common.jl")
end

using LinearAlgebra
using Test

function reference_residual(x, coordinates, p)
    N = length(coordinates)
    output = similar(x, N, N, 2)
    brusselator_2d_reference!(output, reshape(x, N, N, 2), coordinates, p)
    return vec(output)
end

# Differentiate the scalar stencil directly to obtain an independent reference.
function reference_jacobian(x, coordinates, p)
    N = length(coordinates)
    M = N^2
    A, _, alpha, dx = p
    diffusion = alpha / dx^2
    J = zeros(eltype(x), 2 * M, 2 * M)
    indices = LinearIndices((N, N))
    for j in 1:N, i in 1:N
        k = indices[i, j]
        u, v = x[k], x[M + k]
        J[k, k] = -4 * diffusion + 2 * u * v - (A + 1)
        J[k, M + k] = u^2
        J[M + k, k] = A - 2 * u * v
        J[M + k, M + k] = -4 * diffusion - u^2
        for neighbor in (
            indices[mod1(i - 1, N), j],
            indices[mod1(i + 1, N), j],
            indices[i, mod1(j - 1, N)],
            indices[i, mod1(j + 1, N)],
        )
            J[k, neighbor] += diffusion
            J[M + k, M + neighbor] += diffusion
        end
    end
    return J
end

function finite_difference_jacobian(x, coordinates, p; epsilon=1.0e-4)
    return mapreduce(hcat, eachindex(x)) do i
        plus, minus = copy(x), copy(x)
        plus[i] += epsilon
        minus[i] -= epsilon
        return (
            reference_residual(plus, coordinates, p) .-
            reference_residual(minus, coordinates, p)
        ) ./ (2 * epsilon)
    end
end

function validate_brusselator(;
    N=4, damping=1.0, ad_options=Reactant.ADOptimizationOptions()
)
    problem = brusselator_problem(N)
    x = vec(problem.u)
    coordinates, p = problem.coordinates, problem.p
    residual = reference_residual(x, coordinates, p)
    jacobian = reference_jacobian(x, coordinates, p)
    correction = lu(jacobian) \ residual
    expected_step = x .- damping .* correction
    compile_options = brusselator_compile_options(ad_options)

    return @testset "Brusselator Newton N=$N AD=$ad_options" begin
        @test residual_vector(x, coordinates, p) ≈ residual atol = 1.0e-10 rtol = 1.0e-10
        @test finite_difference_jacobian(x, coordinates, p) ≈ jacobian atol = 2.0e-5 rtol =
            2.0e-7

        for (name, fn, host_args, expected) in (
            ("residual", residual_vector, (x, coordinates, p), residual),
            ("jacobian", residual_jacobian, (x, coordinates, p), jacobian),
            ("newton_step", newton_step, (x, coordinates, p, damping), expected_step),
        )
            @testset "$name" begin
                args = Reactant.to_rarray(host_args)
                compiled = Reactant.compile(fn, args; compile_options)
                actual = Array(compiled(args...))
                @test size(actual) == size(expected)
                @test all(isfinite, actual)
                @test actual ≈ expected atol = 1.0e-7 rtol = 1.0e-8
                @test Array(args[1]) == x
                @test Array(args[2]) == coordinates
                if name == "newton_step"
                    delta = (x .- actual) ./ damping
                    scale = norm(jacobian) * norm(delta) + norm(residual)
                    @test norm(jacobian * delta - residual) <= 1.0e-10 * scale
                end
                @info "Validated $name" N ad_options maximum_absolute_error = maximum(
                    abs, actual .- expected
                )
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    options = parse_command_line(ARGS; default_mode="validation")
    options.mode == "validation" ||
        throw(ArgumentError("use runbenchmarks.jl for performance measurements"))
    select_backend(options.backend)
    for (_, ad_options) in options.configurations
        validate_brusselator(; N=options.N, damping=options.damping, ad_options)
    end
end
