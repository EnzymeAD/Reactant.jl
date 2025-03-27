using OffsetArrays: Origin
@testset "Origin" begin
    get_origin(A::AbstractArray) = first.(axes(A))

    @test Origin(0) != Origin((0, ))
    @test Origin(CartesianIndex(1, 2)) === Origin((1, 2)) === Origin(1, 2)

    @test Origin(Int32.((1,2))) == Origin(Int64.((1,2)))
    @test Origin(Int32.((1,2))...) == Origin(Int64.((1,2))...) == Origin((1.0, 2.0))
    @test Origin(Int32(1)) == Origin(Int64(1)) == Origin(1.0)
    @test_throws Exception Origin(1.5)

    # 0d
    A = OffsetArray(zeros())
    B = OffsetArray(zeros(), Origin())
    @test axes(A) == axes(B)

    # 1d
    v = [1, 2]
    @test get_origin(OffsetArray(v, Origin(2))) == (2, )
    ov = OffsetArray(v, -3)
    @test get_origin(OffsetArray(ov, Origin(2))) == (2, )
    @test get_origin(OffsetVector(ov, Origin(2))) == (2, )
    @test get_origin(OffsetArray(ov, Origin((2, )))) == (2, )

    # 2d
    a = [1 2;3 4]
    @test get_origin(OffsetArray(a, Origin(0))) == (0, 0)
    oa = OffsetArray(a, -3, -3)
    @test get_origin(OffsetArray(oa, Origin(0))) == (0, 0)
    @test get_origin(OffsetMatrix(oa, Origin(0))) == (0, 0)
    @test get_origin(OffsetArray(oa, Origin(1, 2))) == (1, 2)

    # 3d
    a = ones(3, 3, 3)
    @test get_origin(OffsetArray(a, Origin(0))) == (0, 0, 0)
    oa = OffsetArray(a, -3, -3, -3)
    @test get_origin(OffsetArray(oa, Origin(0))) == (0, 0, 0)
    @test get_origin(OffsetArray(oa, Origin(1, 2, 3))) == (1, 2, 3)

    # Scalar broadcasting
    let
        a = [ [1,2,3], [4,5,6] ]
        oa = OffsetVector.(a, Origin(0))
        @test get_origin.(oa) == [ (0,), (0,) ]

        a = [ [1 2; 3 4], [5 6 7; 8 9 10] ]
        oa = OffsetArray.(a, Origin(0, -1))
        @test get_origin.(oa) == [ (0,-1), (0,-1) ]
    end

    @testset "as a callable" begin
        a = [1 2; 3 4];
        @test OffsetArray(a, Origin(2)) == Origin(2)(a)
        for (index, firstinds) in Any[(1, (1,1)), ((2,3), (2,3))]
            b = Origin(index)(a)
            @test first.(axes(b)) == firstinds
            @test Origin(b) == Origin(firstinds)
            @test Origin(OffsetArrays.no_offset_view(b)) == Origin(ntuple(_ -> 1, Val(ndims(b))))
        end
        # compatibility with other array types
        @test Origin(Ones(2,2)) == Origin(1,1)
        @test Origin(SMatrix{2,2,Int,4}(1,2,3,4)) == Origin(1,1)
    end
    @testset "display" begin
        io = IOBuffer()
        show(io, Origin(1))
        @test String(take!(io)) == "Origin(1)"
        show(io, Origin(1, 1))
        @test String(take!(io)) == "Origin(1, 1)"
    end

    @testset "avoid overflow (issue #279)" begin
        A = Origin(typemin(Int)+1)(rand(3,3))
        B = Origin(typemax(Int)-4)(A)
        @test first.(axes(B)) == ntuple(_ -> typemax(Int)-4, Val(ndims(B)))
    end
end
