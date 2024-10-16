using BenchmarkTools
using Reactant
using Enzyme
using Boltz, Lux, Random

const SUITE = BenchmarkGroup()

SUITE["runtime"] = BenchmarkGroup()
SUITE["comptime"] = BenchmarkGroup()

SUITE["comptime"]["basics"] = BenchmarkGroup()

SUITE["runtime"]["lux neural networks"] = BenchmarkGroup()
SUITE["comptime"]["lux neural networks"] = BenchmarkGroup()

bcast_cos(x) = cos.(x)

sumcos(x) = sum(cos.(x))

function grad_ip(x)
    dx = Enzyme.make_zero(x)
    Enzyme.autodiff(Reverse, sumcos, Active, Duplicated(x, dx))
    return dx
end

for opt_pass in [:all, :only_enzyme, :before_enzyme, :after_enzyme]
    SUITE["comptime"]["basics"]["2D sum (optimize=$(opt_pass))"] = @benchmarkable begin
        @compile optimize = $(opt_pass) sum(a)
    end setup = begin
        a = Reactant.ConcreteRArray(ones(2, 10))
    end

    SUITE["comptime"]["basics"]["cos.(x) (optimize=$(opt_pass))"] = @benchmarkable begin
        @compile optimize = $(opt_pass) bcast_cos(a)
    end setup = begin
        a = Reactant.ConcreteRArray(ones(2, 10))
    end

    for depth in [11, 13, 16, 19], batchnorm in [false, true]
        SUITE["comptime"]["lux neural networks"]["vgg$(depth) bn=$(batchnorm) (optimize=$(opt_pass))"] = @benchmarkable begin
            @compile optimize = $(opt_pass) vgg(x, ps_concrete, st_concrete)
        end setup = begin
            vgg = Vision.VGG($depth; pretrained=false, batchnorm=$(batchnorm))
            ps, st = Lux.setup(Random.default_rng(), vgg)
            ps_concrete = Reactant.to_rarray(ps)
            st_concrete = Reactant.to_rarray(Lux.testmode(st))
            x = Reactant.to_rarray(rand(Float32, 224, 224, 3, 16))
        end

        SUITE["runtime"]["lux neural networks"]["vgg$(depth) bn=$(batchnorm) (optimize=$(opt_pass))"] = @benchmarkable begin
            vgg_compiled(x, ps_concrete, st_concrete)
        end setup = begin
            vgg = Vision.VGG($depth; pretrained=false, batchnorm=$(batchnorm))
            ps, st = Lux.setup(Random.default_rng(), vgg)
            ps_concrete = Reactant.to_rarray(ps)
            st_concrete = Reactant.to_rarray(Lux.testmode(st))
            x = Reactant.to_rarray(rand(Float32, 224, 224, 3, 16))
            vgg_compiled = @compile optimize = $(opt_pass) vgg(x, ps_concrete, st_concrete)
        end
    end

    for version in (:tiny, :base)
        SUITE["comptime"]["lux neural networks"]["ViT $(version) (optimize=$(opt_pass))"] = @benchmarkable begin
            @compile optimize = $(opt_pass) vit(x, ps_concrete, st_concrete)
        end setup = begin
            vit = Vision.ViT($(Meta.quot(version)))
            ps, st = Lux.setup(Random.default_rng(), vit)
            ps_concrete = Reactant.to_rarray(ps)
            st_concrete = Reactant.to_rarray(Lux.testmode(st))
            x = Reactant.to_rarray(rand(Float32, 256, 256, 3, 16))
        end

        SUITE["runtime"]["lux neural networks"]["ViT $(version) (optimize=$(opt_pass))"] = @benchmarkable begin
            vit_compiled(x, ps_concrete, st_concrete)
        end setup = begin
            vit = Vision.ViT($(Meta.quot(version)))
            ps, st = Lux.setup(Random.default_rng(), vit)
            ps_concrete = Reactant.to_rarray(ps)
            st_concrete = Reactant.to_rarray(Lux.testmode(st))
            x = Reactant.to_rarray(rand(Float32, 256, 256, 3, 16))
            vit_compiled = @compile optimize = $(opt_pass) vit(x, ps_concrete, st_concrete)
        end
    end

    SUITE["comptime"]["basics"]["∇cos (optimize=$(opt_pass))"] = @benchmarkable begin
        @compile optimize = $(opt_pass) grad_ip(a)
    end setup = begin
        a = Reactant.ConcreteRArray(ones(3, 2))
    end
end
