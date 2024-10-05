using Reactant
using Test
using Enzyme
using Statistics

# Reactant.set_default_backend("gpu")

fastmax(x::AbstractArray{T}) where {T} = reduce(max, x; dims=1, init=float(T)(-Inf))

using InteractiveUtils

@testset "2D sum" begin
    x = rand(2, 10)

    r_res = sum(x)

    a = Reactant.ConcreteRArray(x)

    c_res = sum(a)
    @test c_res ≈ r_res

    f = @compile sum(a)

    f_res = f(a)

    @test f_res ≈ r_res
end

@testset "Basic reduce max" begin
    x = rand(2, 10)

    r_res = fastmax(x)

    a = Reactant.ConcreteRArray(x)

    c_res = fastmax(a)
    @test c_res ≈ r_res

    f = @compile fastmax(a)

    f_res = f(a)

    @test f_res ≈ r_res
end

sinexp(x) = sin(exp(x))
sinexpbc(x) = sinexp.(x)

@testset "Broadcast combined" begin
    x = rand(2, 10)

    r_res = sinexpbc(x)

    a = Reactant.ConcreteRArray(x)

    c_res = sinexpbc(a)
    @test c_res ≈ r_res

    f = @compile sinexpbc(a)

    f_res = f(a)

    @test f_res ≈ r_res
end

sumexp(x) = sum(exp, x)

@testset "Basic mapreduce" begin
    x = rand(Float32, 10)
    a = Reactant.ConcreteRArray(x)
    r_res = sumexp(x)

    f = @compile sumexp(a)
    f_res = f(a)

    @test f_res ≈ r_res
end

function mysoftmax!(x)
    max_ = fastmax(x)
    return x .- max_
end

@testset "Basic softmax" begin
    x = rand(2, 10)
    r_res = mysoftmax!(x)

    a = Reactant.ConcreteRArray(x)

    f = @compile mysoftmax!(a)

    f_res = f(a)

    @test f_res ≈ r_res
end

bcast_cos(x) = cos.(x)

@testset "Basic cos" begin
    x = rand(3, 2)
    c = Reactant.ConcreteRArray(x)

    f = @compile bcast_cos(c)
    r = f(c)
    @test r ≈ cos.(x)
end

f_var(args...) = sum(args)

@testset "Vararg" begin
    x = Reactant.to_rarray(ones(3))
    y = Reactant.to_rarray(3 * ones(3))
    z = Reactant.to_rarray(2.6 * ones(3))

    f2 = @compile f_var(x, y, z)
    @test f2(x, y, z) ≈ [6.6, 6.6, 6.6]
end

function sumcos(x)
    return sum(cos.(x))
end

function grad_ip(x)
    dx = Enzyme.make_zero(x)
    Enzyme.autodiff(Reverse, sumcos, Active, Duplicated(x, dx))
    return dx
end

function resgrad_ip(x)
    dx = Enzyme.make_zero(x)
    res = Enzyme.autodiff(ReverseWithPrimal, sumcos, Active, Duplicated(x, dx))
    return (res, dx)
end

@testset "Basic grad cos" begin
    c = Reactant.ConcreteRArray(ones(3, 2))

    f = @compile grad_ip(c)
    r = f(c)

    @test r ≈ -sin.(ones(3, 2))

    f = @compile resgrad_ip(c)
    orig, r = f(c)

    @test orig[2] ≈ sum(cos.(ones(3, 2)))
    @test r ≈ -sin.(ones(3, 2))
end

function mul(A, B)
    return A * B
end
@testset "matmul" begin
    c = Reactant.ConcreteRArray(ones(50, 70))
    d = Reactant.ConcreteRArray(ones(70, 30))

    f = @compile mul(c, d)
    r = f(c, d)

    @test r ≈ mul(ones(50, 70), ones(70, 30))
end

@testset "ConcreteRArray" begin
    c = Reactant.ConcreteRArray(ones(50, 70))
    similar(c)
end

@testset "Reactant.@code_hlo" begin
    W = Reactant.ConcreteRArray(randn(Float32, 10, 20))
    x = Reactant.ConcreteRArray(randn(Float32, 20, 5))
    res = Reactant.@code_hlo W * x
    res_repr = sprint(show, res)

    @test contains(res_repr, "stablehlo.dot_general")
end

@testset "Statistics: `mean` & `var`" begin
    x = randn(2, 3, 4)
    x_ca = Reactant.ConcreteRArray(x)

    mean_fn1(x) = mean(x)
    mean_fn2(x) = mean(x; dims=1)
    mean_fn3(x) = mean(x; dims=(1, 2))
    mean_fn4(x) = mean(x; dims=(1, 3))

    mean_fn1_compiled = @compile mean_fn1(x_ca)
    mean_fn2_compiled = @compile mean_fn2(x_ca)
    mean_fn3_compiled = @compile mean_fn3(x_ca)
    mean_fn4_compiled = @compile mean_fn4(x_ca)

    @test mean_fn1(x) ≈ mean_fn1_compiled(x_ca)
    @test mean_fn2(x) ≈ mean_fn2_compiled(x_ca)
    @test mean_fn3(x) ≈ mean_fn3_compiled(x_ca)
    @test mean_fn4(x) ≈ mean_fn4_compiled(x_ca)

    var_fn1(x) = var(x)
    var_fn2(x) = var(x; dims=1)
    var_fn3(x) = var(x; dims=(1, 2), corrected=false)
    var_fn4(x) = var(x; dims=(1, 3), corrected=false)

    var_fn1_compiled = @compile var_fn1(x_ca)
    var_fn2_compiled = @compile var_fn2(x_ca)
    var_fn3_compiled = @compile var_fn3(x_ca)
    var_fn4_compiled = @compile var_fn4(x_ca)

    @test var_fn1(x) ≈ var_fn1_compiled(x_ca)
    @test var_fn2(x) ≈ var_fn2_compiled(x_ca)
    @test var_fn3(x) ≈ var_fn3_compiled(x_ca)
    @test var_fn4(x) ≈ var_fn4_compiled(x_ca)
end

@testset "concatenation" begin
    @testset "0-dim" begin
        x = fill(true)
        x_concrete = Reactant.to_rarray(x)

        # NOTE [,,,] is a call to `vect`, not `*cat`
        # f = Reactant.compile((x_concrete,)) do x
        #     return [x, x, x]
        # end
        # @test f(x_concrete) ≈ ones(3)

        # vcat
        f = Reactant.compile((x_concrete,)) do x
            return [x; x; x]
        end
        @test f(x_concrete) == ones(Bool, 3)
        @test eltype(f(x_concrete)) === Bool

        # hcat
        f = Reactant.compile((x_concrete,)) do x
            return [x x x]
        end
        @test f(x_concrete) == ones(Bool, 1, 3)
        @test eltype(f(x_concrete)) === Bool

        # hvcat
        f = Reactant.compile((x_concrete,)) do x
            return [x x x; x x x]
        end
        @test f(x_concrete) == ones(Bool, 2, 3)
        @test eltype(f(x_concrete)) === Bool

        # hvncat
        f = Reactant.compile((x_concrete,)) do x
            return [x x x; x x x;;; x x x; x x x]
        end
        @test f(x_concrete) == ones(Bool, 2, 3, 2)
        @test eltype(f(x_concrete)) === Bool

        # typed_vcat
        f = Reactant.compile((x_concrete,)) do x
            return Int[x; x; x]
        end
        @test f(x_concrete) == ones(Int, 3)
        @test eltype(f(x_concrete)) === Int

        # typed_hcat
        f = Reactant.compile((x_concrete,)) do x
            return Int[x; x; x]
        end
        @test f(x_concrete) == ones(Int, 1, 3)
        @test eltype(f(x_concrete)) === Int

        # typed_hvcat
        f = Reactant.compile((x_concrete,)) do x
            return Int[x x x; x x x]
        end
        @test f(x_concrete) == ones(Int, 2, 3)
        @test eltype(f(x_concrete)) === Int

        # typed_hvncat
        f = Reactant.compile((x_concrete,)) do x
            return Int[x x x; x x x;;; x x x; x x x]
        end
        @test f(x_concrete) == ones(Int, 2, 3, 2)
        @test eltype(f(x_concrete)) === Int
    end

    @testset "1-dim" begin
        x = ones(Bool, 2)
        x_concrete = Reactant.to_rarray(x)

        # NOTE [,,,] is a call to `vect`, not `*cat`
        # f = Reactant.compile((x_concrete,)) do x
        #     return [x, x, x]
        # end
        # @test f(x_concrete) ≈ ones(3)

        # vcat
        f = Reactant.compile((x_concrete,)) do x
            return [x; x; x]
        end
        @test f(x_concrete) == ones(Bool, 6)
        @test eltype(f(x_concrete)) === Bool

        # hcat
        f = Reactant.compile((x_concrete,)) do x
            return [x x x]
        end
        @test f(x_concrete) == ones(Bool, 2, 3)
        @test eltype(f(x_concrete)) === Bool

        # hvcat
        f = Reactant.compile((x_concrete,)) do x
            return [x x x; x x x]
        end
        @test f(x_concrete) == ones(Bool, 4, 3)
        @test eltype(f(x_concrete)) === Bool

        # typed_vcat
        f = Reactant.compile((x_concrete,)) do x
            return Int[x; x; x]
        end
        @test f(x_concrete) == ones(Int, 6)
        @test eltype(f(x_concrete)) === Int

        # typed_hcat
        f = Reactant.compile((x_concrete,)) do x
            return Int[x; x; x]
        end
        @test f(x_concrete) == ones(Int, 2, 3)
        @test eltype(f(x_concrete)) === Int

        # typed_hvcat
        f = Reactant.compile((x_concrete,)) do x
            return Int[x x x; x x x]
        end
        @test f(x_concrete) == ones(Int, 4, 3)
        @test eltype(f(x_concrete)) === Int
    end

    x = ones(2, 4, 3)
    x_concrete = Reactant.to_rarray(x)

    cat1(x) = vcat(x, x, x)
    cat2(x) = hcat(x, x, x)
    cat3(x) = cat(x, x, x; dims=Val(3))

    cat1_compiled = @compile cat1(x_concrete)
    cat2_compiled = @compile cat2(x_concrete)
    cat3_compiled = @compile cat3(x_concrete)

    @test cat1(x) ≈ cat1_compiled(x_concrete)
    @test cat2(x) ≈ cat2_compiled(x_concrete)
    @test cat3(x) ≈ cat3_compiled(x_concrete)
end

function update_on_copy(x)
    y = x[1:2, 2:4, :]
    y[1:1, 1:1, :] = ones(1, 1, 3)
    return y
end

@testset "view / setindex" begin
    x = rand(2, 4, 3)
    y = copy(x)
    x_concrete = Reactant.to_rarray(x)
    y_concrete = Reactant.to_rarray(y)

    update_on_copy_compiled = @compile update_on_copy(x_concrete)

    y1 = update_on_copy(x)
    y2 = update_on_copy_compiled(x_concrete)
    @test x == y
    @test x_concrete == y_concrete
    @test y1 == y2

    # function update_inplace(x)
    #     y = view(x, 1:2, 1:2, :)
    #     y[1, 1, :] .= 1
    #     return y
    # end

    # get_indices(x) = x[1:2, 1:2, :]
    # get_view(x) = view(x, 1:2, 1:2, :)

    # get_indices_compiled = @compile get_indices(x_concrete)
    # get_view_compiled = @compile get_view(x_concrete)
end

tuple_byref(x) = (; a=(; b=x))
tuple_byref2(x) = abs2.(x), tuple_byref2(x)

@testset "Tuple byref" begin
    x = Reactant.to_rarray([1.0 -2.0; -3.0 4.0])
    f1 = @compile tuple_byref(x)
    r1 = f1(x)
    @test r1.a.b.data === x.data

    # TODO this seems to hang during compile
    # f2 = @compile tuple_byref2(x)
    # r2 = f2(x)
    # @test r2[2].a.b.data === x.data
    # @test r2[1] == abs2.([1.0 -2.0; -3.0 4.0])
end

sum_xxᵀ(x) = sum(x .* x')

@testset "sum(x .* x')" begin
    @testset "size(x): $(size(x))" for x in (rand(4, 4), rand(4))
        x_ca = Reactant.to_rarray(x)

        sum_xxᵀ_compiled = @compile sum_xxᵀ(x_ca)

        @test sum_xxᵀ_compiled(x_ca) ≈ sum_xxᵀ(x)
    end
end
