using Reactant, Test

function view_getindex_1(x)
    x = view(x, 2:3, 1:2, :)
    return x[2, 1, 1]
end

function view_getindex_2(x)
    x = view(x, 2:3, 1:2, :)
    return x[1:1, 1, :]
end

function view_getindex_3(x)
    x = view(x, 2:3, 1:2, :)
    x2 = view(x, 1:1, 2:2, 1:2)
    return x2[1, 1, 1:1]
end

@testset "view getindex" begin
    x = rand(4, 4, 3)
    x_ra = Reactant.to_rarray(x)

    view_getindex_1_compiled = @compile view_getindex_1(x_ra)

    @test view_getindex_1_compiled(x_ra) ≈ view_getindex_1(x)

    view_getindex_2_compiled = @compile view_getindex_2(x_ra)

    @test view_getindex_2_compiled(x_ra) ≈ view_getindex_2(x)

    view_getindex_3_compiled = @compile view_getindex_3(x_ra)

    @test view_getindex_3_compiled(x_ra) ≈ view_getindex_3(x)
end
