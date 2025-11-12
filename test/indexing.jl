using LinearAlgebra, Reactant, Test

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
    @test contains(repr(hlo), "dynamic_update_slice")

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
    @test contains(repr(hlo), "stablehlo.slice")
    @test @jit(getindex(x_ra, 1:2:4, 3:2:6)) ≈ getindex(x, 1:2:4, 3:2:6)

    hlo = @code_hlo optimize = false getindex(x_ra, 1:3:8)
    @test contains(repr(hlo), "stablehlo.slice")
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
