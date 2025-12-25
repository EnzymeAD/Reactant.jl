using Test
using DLPack, CUDA, Reactant, PyCall
# import Pkg
# Pkg.activate(".")
using Reactant,PyCall,Revise,DLPack,CUDA
# Reactant.set_default_backend("gpu")

includet("./dlpack.jl")
@testset verbose = true "pycall and pytorch" begin
    torch = pyimport("torch")
    for device in ("cpu", "cuda")
        @testset verbose = true "device $device" begin
            @testset verbose = true "pytorch to reactant" begin
                Reactant.set_default_backend(let 
                                                if device=="cpu"
                                                    "cpu"
                                                elseif device=="cuda"
                                                    "gpu"
                                                else
                                                    throw("device $device not implemented")
                                                end
                                            end)
                xt = torch.zeros(3, 5, 4; device=device)
                xt2 = torch.tensor(ones(3, 5, 4); device=device)
                for i in 1:3
                    for j in 1:5
                        for k in 1:4
                            xt[i, j, k] = i + (j - 1) * 3 + (k - 1) * 3 * 5
                            xt2[i, j, k] = i + (j - 1) * 3 + (k - 1) * 3 * 5
                        end
                    end
                end
                xr = from_dlpack(Reactant.ConcretePJRTArray, xt)
                xr2 = from_dlpack(Reactant.ConcretePJRTArray, xt2)
                @testset "correct indices sizes and strides" begin
                    @test Tuple([xt.size()...]) == size(xr)
                    @test Tuple([xt2.size()...]) == size(xr2)
                    @test Tuple([xt.stride()...]) == strides(xr)
                    @test Tuple([xt2.stride()...]) == strides(xr2)
                    # @
                    @test all(@allowscalar xr[i, j, k] == xt[i, j, k].item() for i in 1:3, j in 1:5, k in
                        1:4)
                    @test all(
                        @allowscalar xr2[i, j, k] == xt2[i, j, k].item() for i in 1:3, j in 1:5, k in 1:4
                    )
                    
                    @test all(
                        @allowscalar xr[i, j, k] == xr[ i + (j - 1) * 3 + (k - 1) * 3 * 5] for i in 1:3, j in 1:5, k in 1:4
                    )
                    @test all(
                        @allowscalar xr2[i, j, k] == xr2[ i + (j - 1) * 3 + (k - 1) * 3 * 5] for i in 1:3, j in 1:5, k in 1:4
                    )
                end

                @testset "is an actual view" begin
                    @allowscalar xr[1:2:end] .= 22
                    xt[1] = 9
                    @test all(@allowscalar xr[i, j, k] == xt[i, j, k].item() for i in 1:3, j in 1:5, k in
                    1:4)
                    @allowscalar xr2[1:2:end] .= 22
                    xt2[1] = 9
                    @test all(@allowscalar xr2[i, j, k] == xt2[i, j, k].item() for i in 1:3, j in 1:5, k in
                    1:4)
                end
            end
            @testset verbose = true "reactant to pytorch" begin
                    xr = Reactant.to_rarray(ones(3,5,4))
                    for i in 1:3
                        for j in 1:5
                            for k in 1:4
                                @allowscalar xr[i, j, k] = i + (j - 1) * 3 + (k - 1) * 3 * 5
                            end
                        end
                    end
                    xt = DLPack.share(xr,torch.from_dlpack)
                    @testset "correct indices sizes and strides" begin
                            @test Tuple([xt.size()...]) == size(xr)
                            @test Tuple([xt.stride()...]) == strides(xr)
                            # @
                            @test all(@allowscalar xr[i, j, k] == xt[i, j, k].item() for i in 1:3, j in 1:5, k in
                                1:4)
                            @test all(
                                @allowscalar xr[i, j, k] == xr[ i + (j - 1) * 3 + (k - 1) * 3 * 5] for i in 1:3, j in 1:5, k in 1:4
                            )
                    end
                    @testset "is an actual view" begin
                        @allowscalar xr[1:2:end] .= 22
                        xt[1] = 9
                        @test all(@allowscalar xr[i, j, k] == xt[i, j, k].item() for i in 1:3, j in 1:5, k in
                        1:4)
                    end
            end
        end
    end
end
@testset verbose = true "to/from julia" begin
    for device in ("cpu", "cuda")
        @testset verbose = true "device $device" begin
            @testset verbose = true "julia to reactant" begin
                Reactant.set_default_backend(let 
                                                if device=="cpu"
                                                    "cpu"
                                                elseif device=="cuda"
                                                    "gpu"
                                                else
                                                    throw("device $device not implemented")
                                                end
                                            end)
                
                
                x_org = if device == "cuda"
                    
                    CuArray(ones(3, 5, 4); )
                elseif device=="cpu"
                    @warn "The CPU test may pass/not pass randomly because the alignment for small arrays is not always to 64 bits"
                    ones(3, 5, 4)
                else
                    throw("device $device not implemented")
                end
                for i in 1:3
                    for j in 1:5
                        for k in 1:4
                            @allowscalar x_org[i, j, k] = i + (j - 1) * 3 + (k - 1) * 3 * 5
                        end
                    end
                end
                xr = from_julia(x_org)
                @testset "correct indices sizes and strides" begin
                    @test size(x_org) == size(xr)
                    @test strides(x_org) == strides(xr)
                    # @
                    @test all(@allowscalar xr[i, j, k] == x_org[i, j, k] for i in 1:3, j in 1:5, k in
                        1:4)
                    
                    @test all(
                        @allowscalar xr[i, j, k] == xr[ i + (j - 1) * 3 + (k - 1) * 3 * 5] for i in 1:3, j in 1:5, k in 1:4
                    )
                end

                @testset "is an actual view" begin
                    @allowscalar xr[1:2:end] .= 22
                    @allowscalar x_org[1,:,:] .= 9
                    
                    @test all(@allowscalar xr[i, j, k] == x_org[i, j, k] for i in 1:3, j in 1:5, k in
                    1:4)
                end
            end
            @testset verbose = true "reactant to julia" begin
                    xr = Reactant.to_rarray(ones(3,5,4))
                    for i in 1:3
                        for j in 1:5
                            for k in 1:4
                                @allowscalar xr[i, j, k] = i + (j - 1) * 3 + (k - 1) * 3 * 5
                            end
                        end
                    end
                    x_julia = to_julia(xr)
                    @test typeof(x_julia) <: (if device=="cuda" 
                    CuArray
                else 
                    Array end)
                    @testset "correct indices sizes and strides" begin
                        @test size(x_julia) == size(xr)
                        @test strides(x_julia) == strides(xr)
                        # @
                        @test all(@allowscalar xr[i, j, k] == x_julia[i, j, k] for i in 1:3, j in 1:5, k in
                            1:4)
                        
                        @test all(
                            @allowscalar xr[i, j, k] == xr[ i + (j - 1) * 3 + (k - 1) * 3 * 5] for i in 1:3, j in 1:5, k in 1:4
                        )
                    end
    
                    @testset "is an actual view" begin
                        @allowscalar xr[1:2:end] .= 22
                        @allowscalar x_julia[1,:,:] .= 9
                        
                        @test all(@allowscalar xr[i, j, k] == x_julia[i, j, k] for i in 1:3, j in 1:5, k in
                        1:4)
                    end
            end
        end
    end
end