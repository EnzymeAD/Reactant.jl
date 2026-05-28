using Reactant
using Test

# Jax (and therefore torchax) on GitHub CI dislikes x86 macOS, matching python.jl.
@static if !Sys.isapple() || Sys.ARCH != :x86_64
    using PythonCall

    const ReactantPythonCallExt = Base.get_extension(Reactant, :ReactantPythonCallExt)

    # Relative error metric: max|diff| / max|ref|, robust to scale.
    function relerr(actual, reference)
        ref = Array(reference)
        denom = maximum(abs, ref)
        denom = denom == 0 ? one(denom) : denom
        return maximum(abs, Array(actual) .- ref) / denom
    end

    # Convert a torch tensor to a Julia Array with matching logical shape.
    torch_to_julia(t) = pyconvert(Array, pyimport("numpy").asarray(t.detach().cpu()))

    torch_available =
        ReactantPythonCallExt !== nothing && ReactantPythonCallExt.TORCH_EXPORT_SUPPORTED[]

    if !torch_available
        @warn "torch/torchax not available; skipping PyTorch import tests."
    end

    @testset "PyTorch import" begin
        if !torch_available
            @test_skip false
        else
            torch = pyimport("torch")
            nn = pyimport("torch.nn")
            np = pyimport("numpy")

            @testset "Eager nn.Linear chain" begin
                torch.manual_seed(0)
                model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
                model.eval()

                # Reactant array shape is the torch shape directly (batch first), no
                # reversal: (batch, features).
                xdata = randn(Float32, 4, 8)
                x_torch = torch.from_numpy(np.asarray(xdata))
                y_native = model(x_torch)

                x_ra = Reactant.to_rarray(xdata)
                y = @jit model(x_ra)

                @test y isa ConcreteRArray{Float32,2}
                @test size(y) == (4, 4)
                @test relerr(y, torch_to_julia(y_native)) < 1.0f-4
            end

            @testset "BatchNorm exercises module_kept_var_idx" begin
                torch.manual_seed(0)
                model = nn.Sequential(nn.Linear(6, 6), nn.BatchNorm1d(6), nn.ReLU())
                model.eval()  # use running stats, not per-batch stats

                xdata = randn(Float32, 5, 6)
                x_torch = torch.from_numpy(np.asarray(xdata))
                y_native = model(x_torch)

                x_ra = Reactant.to_rarray(xdata)
                y = @jit model(x_ra)

                @test relerr(y, torch_to_julia(y_native)) < 1.0f-3
            end

            @testset "BatchNorm + lifted constant (state ordering)" begin
                # Conv+BatchNorm with a literal constant in forward. After
                # torch.export's decompositions the lifted constant is ordered before
                # the BatchNorm running-stat buffers in the graph placeholders, which
                # diverges from torchax's default (params, buffers, then constants)
                # state order. Without the input_specs state-ordering fix the constant
                # is bound to a running-stat slot and the result is corrupted. This
                # pattern (conv net + BatchNorm + any constant) is common.
                torch.manual_seed(0)
                pyexec(
                    """
import torch, torch.nn as nn
class _ConvBNConst(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 4, 3)
        self.bn = nn.BatchNorm2d(4)
        self.c2 = nn.Conv2d(4, 2, 3)
    def forward(self, x):
        return self.c2(torch.relu(self.bn(self.c1(x))) + torch.tensor(0.5))
""",
                    @__MODULE__,
                )
                model = pyeval("_ConvBNConst", @__MODULE__)()
                model.eval()

                xdata = randn(Float32, 1, 3, 16, 16)  # NCHW
                y_native = model(torch.from_numpy(np.asarray(xdata)))

                y = @jit model(Reactant.to_rarray(xdata))

                @test size(y) == (1, 2, 12, 12)
                @test relerr(y, torch_to_julia(y_native)) < 1.0f-3
            end

            @testset "TorchScript Conv2d (trace)" begin
                torch.manual_seed(0)
                conv = nn.Conv2d(3, 4, 3)
                conv.eval()
                example = torch.randn(1, 3, 8, 8)
                scripted = torch.jit.trace(conv, example)

                xdata = randn(Float32, 1, 3, 8, 8)  # NCHW, torch order
                x_torch = torch.from_numpy(np.asarray(xdata))
                y_native = scripted(x_torch)

                x_ra = Reactant.to_rarray(xdata)
                y = @jit scripted(x_ra)

                @test size(y) == (1, 4, 6, 6)
                @test relerr(y, torch_to_julia(y_native)) < 1.0f-3
            end
        end
    end
end
