using LinearAlgebra, Reactant, Test, FileCheck

function update_on_copy(x)
    y = x[1:2, 2:4, :]
    y[1:1, 1:1, :] = ones(1, 1, 3)
    return y
end

@testset "view / setindex" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 2, 4, 3)
    y = copy(x)
    x_concrete = Reactant.to_rarray(x)
    y_concrete = Reactant.to_rarray(y)

    y1 = update_on_copy(x)
    y2 = @jit update_on_copy(x_concrete)
    @test x == y
    @test x_concrete == y_concrete
    @test y1 ≈ y2

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

@testset "setindex: unitrange index" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 4, 4)
    y = zeros(3, 2)
    x_ra = Reactant.to_rarray(x)
    y_ra = Reactant.to_rarray(y)

    hlo = @code_hlo optimize = false setindex!(x_ra, y_ra, 6:11)
    @test @filecheck begin
        @check "dynamic_update_slice"
        repr(hlo)
    end

    @jit setindex!(x_ra, y_ra, 6:11)
    setindex!(x, y, 6:11)

    @test x ≈ Array(x_ra)
    @test y ≈ Array(y_ra)
end

function maskset!(y, x)
    y[:] = x
    return nothing
end

@testset "setindex! with vectors & colon indexing" begin
    x = Reactant.to_rarray([4.0])
    y = Reactant.to_rarray([2.0])
    @jit(maskset!(y, x))
    @test y ≈ x

    x = Reactant.to_rarray(ones(3))
    y = Reactant.to_rarray(2 * ones(3))
    @jit(maskset!(y, x))
    @test y ≈ x
end

function masking(x)
    y = similar(x)
    y[1:2, :] .= 0
    y[3:4, :] .= 1
    return y
end

function masking!(x)
    x[1:2, :] .= 0
    x[3:4, :] .= 1
    return x
end

@testset "setindex! with views" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 4, 4) .+ 2.0
    x_ra = Reactant.to_rarray(x)

    y = masking(x)
    y_ra = @jit(masking(x_ra))
    @test y ≈ y_ra

    x_ra_array = Array(x_ra)
    @test !(any(iszero, x_ra_array[1, :]))
    @test !(any(iszero, x_ra_array[2, :]))
    @test !(any(isone, x_ra_array[3, :]))
    @test !(any(isone, x_ra_array[4, :]))

    y_ra = @jit(masking!(x_ra))
    @test y ≈ y_ra

    x_ra_array = Array(x_ra)
    @test @allowscalar all(iszero, x_ra_array[1, :])
    @test @allowscalar all(iszero, x_ra_array[2, :])
    @test @allowscalar all(isone, x_ra_array[3, :])
    @test @allowscalar all(isone, x_ra_array[4, :])
end

function non_contiguous_setindex!(x)
    x[[1, 3, 2], [1, 2, 3, 4]] .= 1.0
    return x
end

@testset "non-contiguous setindex!" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 6, 6)
    x_ra = Reactant.to_rarray(x)

    y = @jit(non_contiguous_setindex!(x_ra))
    y = Array(y)
    x_ra = Array(x_ra)
    @test all(isone, y[1:3, 1:4])
    @test all(isone, x_ra[1:3, 1:4])
    @test !all(isone, y[4:end, :])
    @test !all(isone, x_ra[4:end, :])
    @test !all(isone, y[:, 5:end])
    @test !all(isone, x_ra[:, 5:end])
end

@testset "dynamic indexing" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 5, 3)
    x_ra = Reactant.to_rarray(x)

    idx = [1, 2, 3]
    idx_ra = Reactant.to_rarray(idx)

    fn(x, idx) = @allowscalar x[idx, :]

    y = @jit(fn(x_ra, idx_ra))
    @test y ≈ x[idx, :]
end

@testset "strided indexing" begin
    x = reshape(collect(1:24), 4, 6)
    x_ra = Reactant.to_rarray(x)

    hlo = @code_hlo optimize = false getindex(x_ra, 1:2:4, 3:2:6)
    @test @filecheck begin
        @check "stablehlo.slice"
        repr(hlo)
    end
    @test @jit(getindex(x_ra, 1:2:4, 3:2:6)) ≈ getindex(x, 1:2:4, 3:2:6)

    hlo = @code_hlo optimize = false getindex(x_ra, 1:3:8)
    @test @filecheck begin
        @check "stablehlo.slice"
        repr(hlo)
    end
    @test @jit(getindex(x_ra, 1:3:8)) ≈ getindex(x, 1:3:8)
end

@testset "non-contiguous indexing" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 4, 4, 3)
    x_ra = Reactant.to_rarray(x)

    non_contiguous_indexing1(x) = x[[1, 3, 2], :, :]
    non_contiguous_indexing2(x) = x[:, [1, 2, 1, 3], [1, 3]]

    @test @jit(non_contiguous_indexing1(x_ra)) ≈ non_contiguous_indexing1(x)
    @test @jit(non_contiguous_indexing2(x_ra)) ≈ non_contiguous_indexing2(x)

    x = Reactant.TestUtils.construct_test_array(Float64, 4, 2)
    x_ra = Reactant.to_rarray(x)

    non_contiguous_indexing3(x) = x[[1, 3, 2], :]
    non_contiguous_indexing4(x) = x[:, [1, 2, 2]]

    @test @jit(non_contiguous_indexing3(x_ra)) ≈ non_contiguous_indexing3(x)
    @test @jit(non_contiguous_indexing4(x_ra)) ≈ non_contiguous_indexing4(x)

    x = Reactant.TestUtils.construct_test_array(Float64, 4, 4, 3)
    x_ra = Reactant.to_rarray(x)

    non_contiguous_indexing1!(x) = x[[1, 3, 2], :, :] .= 2
    non_contiguous_indexing2!(x) = x[:, [1, 2, 1, 3], [1, 3]] .= 2

    @jit(non_contiguous_indexing1!(x_ra))
    non_contiguous_indexing1!(x)
    @test x_ra ≈ x

    x = Reactant.TestUtils.construct_test_array(Float64, 4, 4, 3)
    x_ra = Reactant.to_rarray(x)

    @jit(non_contiguous_indexing2!(x_ra))
    non_contiguous_indexing2!(x)
    @test x_ra ≈ x

    x = Reactant.TestUtils.construct_test_array(Float64, 4, 2)
    x_ra = Reactant.to_rarray(x)

    non_contiguous_indexing3!(x) = x[[1, 3, 2], :] .= 2
    non_contiguous_indexing4!(x) = x[:, [1, 2, 2]] .= 2

    @jit(non_contiguous_indexing3!(x_ra))
    non_contiguous_indexing3!(x)
    @test x_ra ≈ x

    x = Reactant.TestUtils.construct_test_array(Float64, 4, 2)
    x_ra = Reactant.to_rarray(x)

    @jit(non_contiguous_indexing4!(x_ra))
    non_contiguous_indexing4!(x)
    @test x_ra ≈ x
end

@testset "indexing with traced arrays" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 4, 4, 3)
    idx1 = [1, 3, 2]
    idx3 = [1, 2, 1, 3]

    x_ra = Reactant.to_rarray(x)
    idx1_ra = Reactant.to_rarray(idx1)
    idx3_ra = Reactant.to_rarray(idx3)

    getindex1(x, idx1) = x[idx1, :, :]
    getindex2(x, idx1) = x[:, idx1, :]
    getindex3(x, idx3) = x[:, :, idx3]
    getindex4(x, idx1, idx3) = x[idx1, :, idx3]

    @test @jit(getindex1(x_ra, idx1_ra)) ≈ getindex1(x, idx1)
    @test @jit(getindex2(x_ra, idx1_ra)) ≈ getindex2(x, idx1)
    @test @jit(getindex3(x_ra, idx3_ra)) ≈ getindex3(x, idx3)
    @test @jit(getindex4(x_ra, idx1_ra, idx3_ra)) ≈ getindex4(x, idx1, idx3)
end

@testset "linear indexing" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 4, 4, 3)
    x_ra = Reactant.to_rarray(x)

    getindex_linear_scalar(x, idx) = @allowscalar x[idx]

    @testset for i in 1:length(x)
        @test @jit(getindex_linear_scalar(x_ra, i)) ≈ getindex_linear_scalar(x, i)
        @test @jit(
            getindex_linear_scalar(x_ra, Reactant.to_rarray(i; track_numbers=Number))
        ) ≈ getindex_linear_scalar(x, i)
    end

    idx = [1, 6, 12, 18, 24, 30, 36, 42]
    idx_ra = Reactant.to_rarray(idx)

    getindex_linear_vector(x, idx) = x[idx]

    @test @jit(getindex_linear_vector(x_ra, idx_ra)) ≈ getindex_linear_vector(x, idx)
    @test @jit(getindex_linear_vector(x_ra, idx)) ≈ getindex_linear_vector(x, idx)

    # indices aren't a TracedRArray but we should still be able to trace
    idx = Reactant.TracedUnitRange(ConcreteRNumber(1), ConcreteRNumber(5), 5)
    @test @jit(getindex_linear_vector(x_ra, idx)) ≈ getindex_linear_vector(x, 1:5)
end

@testset "Boolean Indexing" begin
    x_ra = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 4, 16))
    idxs = [iszero(mod(i, 3)) for i in 1:16]
    idxs_ra = Reactant.to_rarray(idxs)

    fn(x, idxs) = x[:, idxs]

    @test_throws ErrorException @jit(fn(x_ra, idxs_ra))

    res = @jit fn(x_ra, Array(idxs_ra))
    @test res ≈ fn(Array(x_ra), Array(idxs_ra))
end

@testset "inconsistent indexing" begin
    x_ra = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float64, 3, 4, 3))
    idx_ra = Reactant.to_rarray(1; track_numbers=Number)

    fn1(x) = x[:, :, 1]
    fn2(x, idx) = x[:, :, idx]
    fn3(x, idx) = x[idx, :, 1]

    @test ndims(@jit(fn1(x_ra))) == 2
    @test ndims(@jit(fn2(x_ra, idx_ra))) == 2
    @test ndims(@jit(fn3(x_ra, idx_ra))) == 1
end

@testset "High-Dimensional Array Indexing" begin
    x_ra = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float64, 5, 4, 3))
    idx1 = reshape(mod1.(collect(1:12), 5), 2, 2, 3)
    idx2 = reshape(mod1.(collect(1:12), 4), 2, 2, 3)
    idx3 = reshape(mod1.(collect(1:12), 3), 2, 2, 3)
    idx1_ra = Reactant.to_rarray(idx1)
    idx2_ra = Reactant.to_rarray(idx2)

    fn(x, idx1, idx2, idx3) = x[idx1, idx2, idx3]

    @test @jit(fn(x_ra, idx1_ra, idx2_ra, idx3)) ≈
        fn(Array(x_ra), Array(idx1_ra), Array(idx2_ra), idx3)
end

function issue_617(outf, fr, pr, I)
    tmp = fr .* reshape(pr, size(fr))
    outv = @view outf[I]
    vtmp = vec(tmp)
    outv .= vtmp
    return outf
end

@testset "issue #617" begin
    N, M = 4, 6

    f = Reactant.TestUtils.construct_test_array(ComplexF32, N, N)
    p = Reactant.TestUtils.construct_test_array(ComplexF32, N * N)
    I = 1:(N^2)
    out = Reactant.TestUtils.construct_test_array(ComplexF32, M, M)

    fr = Reactant.to_rarray(f)
    pr = Reactant.to_rarray(p)
    outr = Reactant.to_rarray(out)
    Ir = Reactant.to_rarray(I)

    @test @jit(issue_617(outr, fr, pr, Ir)) ≈ issue_617(out, f, p, I)
end

function scalar_setindex(x, idx, val)
    @allowscalar x[idx] = val
    return x
end

@testset "scalar setindex" begin
    x = zeros(4, 4)
    x_ra = Reactant.to_rarray(x)

    @test @jit(scalar_setindex(x_ra, 1, 1)) ≈ scalar_setindex(x, 1, 1)
    @test @allowscalar x_ra[1] == 1

    x = zeros(4, 4)
    x_ra = Reactant.to_rarray(x)

    @test @jit(scalar_setindex(x_ra, ConcreteRNumber(1), 1)) ≈ scalar_setindex(x, 1, 1)
    @test @allowscalar x_ra[1] == 1
end

function write_with_broadcast1!(x, y)
    x[1, :, :] .= reshape(y, 4, 3)
    return x
end
function write_with_broadcast2!(x, y)
    x[:, 1, :] .= view(y, :, 1:3)
    return x
end

@testset "write_with_broadcast" begin
    x_ra = Reactant.to_rarray(zeros(3, 4, 3))
    y_ra = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float64, 3, 4))

    res = @jit write_with_broadcast1!(x_ra, y_ra)

    if res.data isa Tuple
        @test res.data[1] === x_ra.data[1]
    else
        @test res.data === x_ra.data
    end

    res = Array(res)
    y = Array(y_ra)
    @test res[1, :, :] ≈ reshape(y, 4, 3)

    x_ra = Reactant.to_rarray(zeros(3, 4, 3))
    y_ra = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float64, 3, 4))

    res = @jit write_with_broadcast2!(x_ra, y_ra)

    if res.data isa Tuple
        @test res.data[1] === x_ra.data[1]
    else
        @test res.data === x_ra.data
    end

    res = Array(res)
    y = Array(y_ra)
    @test res[:, 1, :] ≈ view(y, :, 1:3)
end

@testset "getindex ambiguity" begin
    x = collect(Float32, 1:8)
    x_ra = Reactant.to_rarray(x)

    idx = CartesianIndex(1)

    fn(x, idx) = @allowscalar x[idx]

    @test @jit(fn(x_ra, idx)) ≈ fn(x, idx)
end

@testset "ConcreteRArray view fill!" begin
    x = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float64, 2, 3))
    x_view = view(x, 1:2, 1:2)

    fill!(x_view, 0.0)
    @test all(Array(x)[1:2, 1:2] .== 0)
end

@testset "ConcreteRArray mapreducedim!" begin
    dest = ones(3, 1)
    x = Reactant.TestUtils.construct_test_array(Float64, 3, 3)
    dest_ra = Reactant.to_rarray(dest)
    x_ra = Reactant.to_rarray(x)

    Base.mapreducedim!(sin, +, dest_ra, x_ra)
    Base.mapreducedim!(sin, +, dest, x)

    @test dest_ra ≈ dest
end

@testset "ConcreteRArray destination view mapreducedim" begin
    parent = ones(3, 1)
    x = Reactant.TestUtils.construct_test_array(Float64, 2, 3)
    parent_ra = Reactant.to_rarray(parent)
    x_ra = Reactant.to_rarray(x)

    dest_ra = @view parent_ra[1:2, 1:1]
    Base.mapreducedim!(sin, +, dest_ra, x_ra)
    dest = @view parent[1:2, 1:1]
    Base.mapreducedim!(sin, +, dest, x)

    @test parent_ra ≈ parent
end

function test_slice_copy!(x)
    @view(x[1, :]) .= 0
    return x
end

@testset "slice copy" begin
    x = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float64, 2, 10))
    @jit test_slice_copy!(x)
    @test all(Array(x)[1, :] .== 0)
end

@testset "reshaped view setindex!" begin
    du = Reactant.TestUtils.construct_test_array(Float32, 10)
    u = Reactant.TestUtils.construct_test_array(Float32, 10)
    p = 16.0f0
    t = 12.0f0

    du_ra = Reactant.to_rarray(du; track_numbers=Number)
    u_ra = Reactant.to_rarray(u; track_numbers=Number)
    p_ra = Reactant.to_rarray(p; track_numbers=Number)
    t_ra = Reactant.to_rarray(t; track_numbers=Number)

    function odef(du, u, p, t)
        u = reshape(u, 2, :)
        du = reshape(du, 2, :)
        @view(du[1, :]) .= @view(u[1, :]) .+ p .+ t
        @view(du[2, :]) .= @view(u[2, :]) .* 2
        return reshape(du, :)
    end

    @test @jit(odef(du_ra, u_ra, p_ra, t_ra)) ≈ odef(du, u, p, t)
end

function index_jl_array(idxs)
    a = collect(1:10)
    return @allowscalar @inbounds a[idxs]
end

function index_jl_array(idx1, idx2)
    a = reshape(collect(1:10), 2, 5)
    return @allowscalar @inbounds a[idx1, idx2]
end

@testset "Indexing JL Array with Traced Arrays" begin
    idxs = [2, 4, 7]
    idxs_ra = Reactant.to_rarray(idxs)
    @test @jit(index_jl_array(idxs_ra)) ≈ index_jl_array(idxs)

    idx1 = [1]
    idx2 = [1, 3, 5]
    idx1_ra = Reactant.to_rarray(idx1)
    idx2_ra = Reactant.to_rarray(idx2)

    @test @jit(index_jl_array(idx1_ra, idx2_ra)) ≈ index_jl_array(idx1, idx2)
    @test @jit(index_jl_array(idx1_ra, 3)) ≈ index_jl_array(idx1, 3)
    @test @jit(index_jl_array(1, idx2_ra)) ≈ index_jl_array(1, idx2)
end

function bcast_setindex!(outr, arr, dindx)
    @allowscalar @inbounds outr[dindx] .= arr
    return nothing
end

@testset "bcast setindex!" begin
    f = rand(ComplexF32, 4, 4)
    p = rand(ComplexF32, 4 * 4)
    fr = Reactant.to_rarray(f)
    pr = Reactant.to_rarray(p)
    dindxr = Reactant.to_rarray(collect(1:length(fr)))

    bcast_setindex!(f, p, collect(1:length(fr)))
    @jit bcast_setindex!(fr, pr, dindxr)
    @test fr ≈ f
end

copyto_with_reshaped_view(out, g, I) = copyto!(out, reshape(@view(g[I]), :))

@testset "copyto! with reshaped view" begin
    outr = Reactant.to_rarray(zeros(6, 6))
    gr = Reactant.to_rarray(ones(6, 6))
    Ir = Reactant.to_rarray(eachindex(1:length(outr)))

    @jit copyto_with_reshaped_view(outr, gr, Ir)
    @test all(isone, Array(outr))
end

function comparison_of_views(a, b)
    view(parent(a), 1:1, 1:1, 1:1) .= view(parent(b), 1:1, 1:1, 1:1)
    return a
end

@testset "comparison of views" begin
    a = zeros(208, 142, 1)
    b = ones(208, 112, 1)

    a_r = Reactant.to_rarray(a)
    b_r = Reactant.to_rarray(b)

    c_r = comparison_of_views(a_r, b_r)
    c = comparison_of_views(a, b)
    @test c_r ≈ c
end

function tracestep(x)
    I = 1:5:20
    s = zero(eltype(x))
    @allowscalar @trace for i in eachindex(I)
        idx = I[i]
        s += x[idx]
    end
    return s
end

@testset "StepRange Traced Indexing" begin
    x = Reactant.to_rarray(collect(Float32, 1:20))
    x_ra = Reactant.to_rarray(x)
    @test @jit(tracestep(x_ra)) ≈ tracestep(x)
end

@testset "Indexing: $(x)" for x in (1:3:10, range(-0.1, 0.3; length=5))
    x_ra = Reactant.to_rarray(x; track_numbers=true)
    @testset "idx: $(idx)" for idx in (1, 2, 3)
        @test @jit(getindex(x_ra, idx)) ≈ getindex(x, idx)

        idx_ra = Reactant.to_rarray(idx; track_numbers=true)
        @test @jit(getindex(x_ra, idx_ra)) ≈ getindex(x, idx)
    end
end

# https://github.com/EnzymeAD/Reactant.jl/issues/3391
function issue3391_columns_2d(x)   # (16,) -> (16, 2)
    P = similar(x, 16, 2)
    view(P, :, 1) .= x
    view(P, :, 2) .= 2 .* x
    return P
end
function issue3391_trailing_3d(x)  # (16, 2) -> (16, 2, 2)
    P = similar(x, 16, 2, 2)
    view(P, :, :, 1) .= x
    view(P, :, :, 2) .= 2 .* x
    return P
end
function issue3391_middle_3d(x)    # (16,) -> (16, 2, 1)
    P = similar(x, 16, 2, 1)
    view(P, :, 1, 1) .= x
    view(P, :, 2, 1) .= 2 .* x
    return P
end
function issue3391_range_view(x)
    P = similar(x, 16, 2)
    P .= 0
    view(P, 2:5, 1) .= x[2:5]
    view(P, 2:5, 2) .= 2 .* x[2:5]
    return P
end
function issue3391_setindex_col(x)
    P = similar(x, 16, 2)
    P .= 0
    P[:, 1] = x
    P[:, 2] .= 2 .* x
    return P
end
function issue3391_row(x)
    P = similar(x, 4, 16)
    P .= 0
    P[3, :] .= x
    return P
end
function issue3391_middle_colon_3d(x)  # (16,) -> (16, 2, 3)
    P = similar(x, 16, 2, 3)
    P .= 0
    view(P, :, 1, :) .= x
    view(P, :, 2, :) .= 2 .* x
    return P
end
function issue3391_nested_view(x)
    P = similar(x, 16, 2)
    P .= 0
    view(view(P, :, 2), 3:6) .= x[1:4]
    return P
end
function issue3391_view_of_reshape(x)
    P = similar(x, 32)
    P .= 0
    view(reshape(P, 16, 2), :, 2) .= x
    return P
end
function issue3391_vector_parent(x)
    P = similar(x, 16)
    P .= 0
    view(P, 3:6) .= x[1:4]
    view(P, 9:12) .+= view(x, 5:8) .* 2
    return P
end

@testset "column setindex! on 1-D views lowers to slices, not scatter (#3391)" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 16)
    x_ra = Reactant.to_rarray(x)
    x2 = Reactant.TestUtils.construct_test_array(Float64, 16, 2)
    x2_ra = Reactant.to_rarray(x2)

    @testset "$(f)" for (f, a, a_ra) in (
        (issue3391_columns_2d, x, x_ra),
        (issue3391_trailing_3d, x2, x2_ra),
        (issue3391_middle_3d, x, x_ra),
        (issue3391_range_view, x, x_ra),
        (issue3391_setindex_col, x, x_ra),
        (issue3391_row, x, x_ra),
        (issue3391_middle_colon_3d, x, x_ra),
        (issue3391_nested_view, x, x_ra),
        (issue3391_view_of_reshape, x, x_ra),
        (issue3391_vector_parent, x, x_ra),
    )
        hlo = repr(@code_hlo f(a_ra))
        @test !occursin("stablehlo.scatter", hlo)
        @test !occursin("stablehlo.gather", hlo)
        @test @jit(f(a_ra)) ≈ f(a)
    end

    # getindex on a 1-D view with a range should be a slice too
    getcol_range(x) = view(x, :, 2)[2:5]
    hlo = repr(@code_hlo getcol_range(x2_ra))
    @test !occursin("stablehlo.gather", hlo)
    @test @jit(getcol_range(x2_ra)) ≈ getcol_range(x2)

    # out-of-bounds indices into a 1-D view still throw at trace time
    getcol_oob(x) = view(x, :, 2)[[1, 3, 17]]
    @test_throws BoundsError @code_hlo getcol_oob(x2_ra)
end

# irregular indices on 1-D views (and the equivalent direct setindex!) still work
function irregular_view_setindex!(x)
    P = similar(x, 4, 4)
    P .= 0
    v = view(P, :, 2)
    v[[1, 3]] .= 7
    v2 = view(P, 2, :)
    v2[[true, false, true, false]] .= x[1:2, 1]
    return P
end
function setindex_dropped_dim_scatter!(x)
    P = similar(x, 4, 4)
    P .= 0
    P[[1, 3], 2] = x[1:2, 1]
    P[4, [1, 3]] .= 5
    return P
end
function setindex_rank_mismatch!(x)
    P = similar(x, 4, 4)
    P .= 0
    P[2, :] = reshape(x[:, 1], 1, 4)
    return P
end

@testset "irregular 1-D view / dropped-dim setindex!" begin
    x = Reactant.TestUtils.construct_test_array(Float64, 4, 4)
    x_ra = Reactant.to_rarray(x)
    @test @jit(irregular_view_setindex!(x_ra)) ≈ irregular_view_setindex!(x)
    @test @jit(setindex_dropped_dim_scatter!(x_ra)) ≈ setindex_dropped_dim_scatter!(x)
    @test @jit(setindex_rank_mismatch!(x_ra)) ≈ setindex_rank_mismatch!(x)

    getview_irregular(x) = view(x, :, 2)[[1, 3, 4]]
    @test @jit(getview_irregular(x_ra)) ≈ getview_irregular(x)
    getview_cartesian(x) = view(x, :, 2)[[CartesianIndex(1), CartesianIndex(4)]]
    @test @jit(getview_cartesian(x_ra)) ≈ getview_cartesian(x)
    getview_mask(x) = view(x, 3, :)[[true, false, true, false]]
    @test @jit(getview_mask(x_ra)) ≈ getview_mask(x)
end

# broadcast assignment through a view with traced integer indices (#2960)
issue2960_assign!(x, idxs, v) = (x[idxs] .= v; x)
issue2960_view!(x, idxs, v) = (view(x, idxs) .= v; x)
issue2960_scalar!(x, idxs, _) = (x[idxs] .= 2.0; x)
issue2960_accum!(x, idxs, v) = (x[idxs] .+= v; x)
issue2960_col!(x, idxs, v) = (x[idxs, 2] .= v; x)
issue2960_row!(x, idxs, v) = (x[2, idxs] .= v; x)
issue2960_rows!(x, idxs, v) = (x[idxs, :] .= v; x)
issue2960_rows_scalar!(x, idxs, _) = (x[idxs, :] .= 3.0; x)

@testset "broadcast setindex! through view with traced indices (#2960)" begin
    x = collect(Float64, 1:10)
    idxs = [1, 3, 4]
    v = [10.0, 20.0, 30.0]
    X = reshape(collect(Float64, 1:20), 5, 4)
    v4 = [1.0, 2.0, 3.0, 4.0]

    @testset "$(f)" for (f, a, i, b) in (
        (issue2960_assign!, x, idxs, v),
        (issue2960_view!, x, idxs, v),
        (issue2960_scalar!, x, idxs, v),
        (issue2960_accum!, x, idxs, v),
        (issue2960_col!, X, idxs, v),
        (issue2960_row!, X, idxs, v),
        (issue2960_rows!, X, idxs, v),
        (issue2960_rows!, X, idxs, collect(v4')),
        (issue2960_rows_scalar!, X, idxs, v),
    )
        a_ra = Reactant.to_rarray(a)
        i_ra = Reactant.to_rarray(i)
        b_ra = Reactant.to_rarray(b)
        hlo = repr(@code_hlo f(a_ra, i_ra, b_ra))
        @test occursin("stablehlo.scatter", hlo)
        @test @jit(f(a_ra, i_ra, b_ra)) ≈ f(copy(a), i, b)
    end

    # the non-broadcast form and reading through traced indices keep working
    issue2960_setindex!(x, idxs, v) = (x[idxs] = v; x)
    x_ra = Reactant.to_rarray(x)
    @test @jit(issue2960_setindex!(x_ra, Reactant.to_rarray(idxs), Reactant.to_rarray(v))) ≈
        issue2960_setindex!(copy(x), idxs, v)
    issue2960_read(x, idxs) = x[idxs] .+ 1
    @test @jit(issue2960_read(Reactant.to_rarray(x), Reactant.to_rarray(idxs))) ≈
        issue2960_read(x, idxs)
end
