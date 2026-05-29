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

            # The import path leaves matmul precision at jax's platform default (see
            # pytorch.jl), which on a GPU-initialized jax lowers float32 matmuls at
            # reduced (TF32-style) precision and diverges from eager torch by ~5e-4.
            # The tests want exact, deterministic comparisons, so force full float32
            # precision here. This is a test-only setting; it does not change the
            # default behavior of the import path for users.
            pyimport("jax").config.update("jax_default_matmul_precision", "highest")

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
                # 1e-4 holds across CPU and GPU because the lowering pins matmul
                # precision to "highest" (see pytorch.jl). Without that pin jax
                # lowers float32 matmuls at reduced precision when it initializes for
                # a GPU platform, pushing the error against eager torch to ~5e-4.
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

                @test relerr(y, torch_to_julia(y_native)) < 1.0f-4
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
                @test relerr(y, torch_to_julia(y_native)) < 1.0f-4
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
                @test relerr(y, torch_to_julia(y_native)) < 1.0f-4
            end

            @testset "Multiple inputs (operand ordering)" begin
                # forward(self, a, b) takes two tensors. Exercises the
                # (inputs..., weights...) reordering and the n_inputs accounting for
                # length(args) > 1, which the single-input tests never hit.
                torch.manual_seed(0)
                pyexec(
                    """
import torch, torch.nn as nn
class _TwoInput(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 4)
    def forward(self, a, b):
        return self.lin(a) + b
""",
                    @__MODULE__,
                )
                model = pyeval("_TwoInput", @__MODULE__)()
                model.eval()

                adata = randn(Float32, 3, 8)
                bdata = randn(Float32, 3, 4)
                y_native = model(
                    torch.from_numpy(np.asarray(adata)), torch.from_numpy(np.asarray(bdata))
                )

                a_ra = Reactant.to_rarray(adata)
                b_ra = Reactant.to_rarray(bdata)
                y = @jit model(a_ra, b_ra)

                @test size(y) == (3, 4)
                @test relerr(y, torch_to_julia(y_native)) < 1.0f-4
            end

            @testset "Multiple outputs (result ordering)" begin
                # forward returns a tuple of two tensors. Exercises the length(res) > 1
                # branch of pycall_with_torch_export and output ordering.
                torch.manual_seed(0)
                pyexec(
                    """
import torch, torch.nn as nn
class _TwoOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 4)
    def forward(self, x):
        h = self.lin(x)
        return h, torch.relu(h)
""",
                    @__MODULE__,
                )
                model = pyeval("_TwoOutput", @__MODULE__)()
                model.eval()

                xdata = randn(Float32, 3, 8)
                native = model(torch.from_numpy(np.asarray(xdata)))
                y0_native = native[0]
                y1_native = native[1]

                x_ra = Reactant.to_rarray(xdata)
                y = @jit model(x_ra)

                @test y isa Tuple
                @test length(y) == 2
                @test size(y[1]) == (3, 4)
                @test size(y[2]) == (3, 4)
                @test relerr(y[1], torch_to_julia(y0_native)) < 1.0f-4
                @test relerr(y[2], torch_to_julia(y1_native)) < 1.0f-4
            end
        end
    end
end
