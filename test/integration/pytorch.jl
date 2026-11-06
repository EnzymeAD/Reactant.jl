using Reactant
using Test

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

torch_available = ReactantPythonCallExt !== nothing && ReactantPythonCallExt._torch_ready()

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

        @testset "Float64 (double precision)" begin
            # Double precision is common in the Julia scientific ecosystem. jax
            # disables 64-bit by default, so the export helper enables it for the
            # lowering (see pytorch.jl); otherwise the model would be downcast to
            # float32 and hlo_call would reject the float64 operands. The result
            # should stay Float64 and match eager torch to near machine epsilon.
            torch.manual_seed(0)
            model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
            model.double()
            model.eval()

            xdata = randn(Float64, 4, 8)
            y_native = model(torch.from_numpy(np.asarray(xdata)))

            y = @jit model(Reactant.to_rarray(xdata))

            @test y isa ConcreteRArray{Float64,2}
            @test size(y) == (4, 4)
            @test relerr(y, torch_to_julia(y_native)) < 1.0e-12
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
            # state order. Without the arguments state-ordering fix the constant
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

        @testset "Weak scalar constant (FP64 promotion under x64)" begin
            # A float32 model whose forward adds a 0-dim float64 scalar tensor.
            # PyTorch's scalar (0-dim) promotion rule keeps the result float32:
            # the float64 scalar binds to its float32 operand instead of
            # upcasting it. The export helper enables jax x64 (see pytorch.jl),
            # under which jax would otherwise keep the constant a strong float64
            # array that upcasts and collides with the float32 operands in strict
            # ops. _demote_weak_scalar_constants demotes the 0-dim float64
            # constant back to float32 so the result stays Float32 and matches
            # eager torch. (Bare Python float literals are constant-folded by
            # torch.export and never reach this path, so use an explicit float64
            # scalar tensor to exercise it.)
            torch.manual_seed(0)
            pyexec(
                """
import torch, torch.nn as nn
class _WeakScalar(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 4)
    def forward(self, x):
        return self.lin(x) + torch.tensor(0.5, dtype=torch.float64)
""",
                @__MODULE__,
            )
            model = pyeval("_WeakScalar", @__MODULE__)()
            model.eval()

            xdata = randn(Float32, 3, 8)
            y_native = model(torch.from_numpy(np.asarray(xdata)))

            @test pyconvert(String, pyimport("builtins").str(y_native.dtype)) ==
                "torch.float32"

            y = @jit model(Reactant.to_rarray(xdata))

            @test y isa ConcreteRArray{Float32,2}
            @test size(y) == (3, 4)
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

        @testset "import_torch: parameters and buffers as arguments" begin
            torch.manual_seed(0)
            model = nn.Sequential(
                nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 4)
            )
            # Give the running statistics non-default values so that passing
            # them through, or reading the wrong one, is visible.
            model[1].running_mean.copy_(torch.randn(16))
            model[1].running_var.copy_(torch.rand(16) + 0.5)
            model.eval()

            tm, ps, st = Reactant.import_torch(model)

            # Flat NamedTuples keyed by FQN, in torch's registration order.
            @test collect(string.(keys(ps))) ==
                ["0.weight", "0.bias", "1.weight", "1.bias", "3.weight", "3.bias"]
            @test collect(string.(keys(st))) ==
                ["1.running_mean", "1.running_var", "1.num_batches_tracked"]
            @test ps.var"0.weight" isa ConcreteRArray{Float32,2}
            @test size(ps.var"0.weight") == (16, 8)
            @test st.var"1.num_batches_tracked" isa ConcreteRNumber{Int64}

            xdata = randn(Float32, 4, 8)
            x_torch = torch.from_numpy(np.asarray(xdata))
            x = Reactant.to_rarray(xdata)

            @testset "layout and FQN access" begin
                sd = model.state_dict()
                layout = Reactant.torch_layout(tm)
                # One entry per tensor, parameters then buffers, in torch's order.
                @test [e.fqn for e in layout] == vcat(
                    [pyconvert(String, kv[0]) for kv in model.named_parameters()],
                    [pyconvert(String, kv[0]) for kv in model.named_buffers()],
                )
                @test [e.fqn for e in layout] == string.([keys(ps)..., keys(st)...])
                @test Set(e.fqn for e in layout) ==
                    Set(pyconvert(String, k) for k in sd.keys())

                e = Reactant.torch_layout(tm, "1.running_mean")
                @test e.kind === :buffer
                @test e.persistent
                @test e.owner == "BatchNorm1d"
                @test e.eltype === Float32 && e.shape == (16,)
                @test Reactant.torch_layout(tm, "0.weight").owner == "Linear"
                @test Reactant.torch_layout(tm, "1.num_batches_tracked").eltype === Int64
                @test_throws KeyError Reactant.torch_layout(tm, "9.weight")

                # Every leaf matches its state_dict tensor and its layout entry.
                for e in layout
                    tree = e.kind === :parameter ? ps : st
                    leaf = tree[Symbol(e.fqn)]
                    @test Reactant.unwrapped_eltype(leaf) === e.eltype
                    @test size(leaf) == e.shape
                    host = leaf isa Number ? fill(convert(e.eltype, leaf)) : Array(leaf)
                    @test host == torch_to_julia(sd[e.fqn])
                end
                @test st[Symbol("1.running_mean")] === st.var"1.running_mean"
                @test !haskey(ps, Symbol("1.running_mean"))  # a buffer, so in st

                s = sprint(show, MIME"text/plain"(), tm)
                @test occursin("parameters (ps), 6 tensors, 244 elements:", s)
                @test occursin("buffers (st), 3 tensors, 33 elements:", s)
                @test occursin(r"^\s+1\.running_mean\s+Float32\s+\(16,\)\s+BatchNorm1d"m, s)
            end

            @testset "eval" begin
                y_native = model(x_torch)
                y, st_new = @jit tm(x, ps, st)
                @test y isa ConcreteRArray{Float32,2}
                @test relerr(y, torch_to_julia(y_native)) < 1.0f-4
                @test relerr(y, @jit(model(x))) < 1.0f-6
                @test keys(st_new) == keys(st)
                @test Array(st_new.var"1.running_mean") == Array(st.var"1.running_mean")
                @test Array(st_new.var"1.running_var") == Array(st.var"1.running_var")
                @test Int(st_new.var"1.num_batches_tracked") == 0
                @test length(tm.cache) == 1
            end

            @testset "parameters are arguments, not constants" begin
                # An embedded 16x8 weight appears as a dense constant, possibly
                # transposed by the optimizer; as an argument it does not.
                weight_const = r"stablehlo\.constant dense.*: tensor<(16x8|8x16)xf32>"
                hlo = sprint(show, @code_hlo tm(x, ps, st))
                @test !occursin(weight_const, hlo)
                # Reactant lists entry arguments in column-major order.
                @test occursin(r"%arg\d+: tensor<(16x8|8x16)xf32>", hlo)
                shortcut = sprint(show, @code_hlo model(x))
                @test occursin(weight_const, shortcut)
            end

            @testset "new parameter values reuse the export" begin
                w = Array(ps.var"3.weight")
                ps2 = merge(ps, (; var"3.weight"=Reactant.to_rarray(2 .* w)))
                y1, _ = @jit tm(x, ps, st)
                y2, _ = @jit tm(x, ps2, st)
                b = Array(ps.var"3.bias")
                @test Array(y2) ≈ 2 .* (Array(y1) .- b') .+ b' rtol = 1.0f-5
                @test length(tm.cache) == 1
            end

            @testset "shape mismatch names the FQN" begin
                bad = merge(ps, (; var"0.weight"=Reactant.to_rarray(zeros(Float32, 16, 7))))
                err = try
                    @jit tm(x, bad, st)
                    nothing
                catch e
                    e
                end
                @test err !== nothing
                @test occursin("\"0.weight\"", sprint(showerror, err))
            end

            @testset "train: buffer updates match eager torch" begin
                model.train()
                y1, st1 = @jit tm(x, ps, st)
                @test length(tm.cache) == 2
                y1_native = model(x_torch)  # updates model's running statistics
                @test relerr(y1, torch_to_julia(y1_native)) < 1.0f-4
                @test relerr(
                    st1.var"1.running_mean", torch_to_julia(model[1].running_mean)
                ) < 1.0f-5
                @test relerr(st1.var"1.running_var", torch_to_julia(model[1].running_var)) <
                    1.0f-5
                @test Int(st1.var"1.num_batches_tracked") == 1
                # Train-mode BatchNorm normalizes with batch statistics, so the
                # output differs from evaluation mode.
                model.eval()
                y_eval, _ = @jit tm(x, ps, st)
                @test !(Array(y_eval) ≈ Array(y1))
                model.train()
                @test length(tm.cache) == 2

                # Feed the returned state back in for a second step.
                x2data = randn(Float32, 4, 8)
                y2, st2 = @jit tm(Reactant.to_rarray(x2data), ps, st1)
                y2_native = model(torch.from_numpy(np.asarray(x2data)))
                @test relerr(y2, torch_to_julia(y2_native)) < 1.0f-4
                @test relerr(
                    st2.var"1.running_mean", torch_to_julia(model[1].running_mean)
                ) < 1.0f-5
                @test relerr(st2.var"1.running_var", torch_to_julia(model[1].running_var)) <
                    1.0f-5
                @test Int(st2.var"1.num_batches_tracked") == 2
                @test st2.var"1.num_batches_tracked" isa ConcreteRNumber{Int64}

                # Exporting must not have changed the module's mode.
                @test pyconvert(Bool, model.training)
                model.eval()
                @test pyconvert(Bool, model[1].training) == false
            end

            @testset "display" begin
                s = sprint(show, MIME"text/plain"(), tm)
                @test occursin("TorchModule(Sequential) with 2 exports", s)
                @test occursin(r"export :train via torchax for inputs Float32\(", s)
                @test occursin(":buffer_mutation", s)
                # Each result names the st leaf it replaces.
                @test occursin(r":buffer_mutation\s+st\.var\"1\.running_mean\""m, s)
                @test occursin(r":user_output\s+y$"m, s)
            end
        end

        @testset "import_torch: buffer mutation (state in, state out)" begin
            # A module that updates buffers in place on every call, independent of
            # train/eval mode, covering: a float buffer updated from the input and
            # then read by the output (`ema`), a 0-dim int64 counter (`steps`), a
            # non-persistent buffer (`scratch`), a buffer in a nested submodule
            # (`inner.total`), a buffer that is read but never written (`scale`)
            # and one that is never read (`unused`).
            pyexec(
                """
import torch, torch.nn as nn
class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("total", torch.zeros(()))
    def forward(self, h):
        self.total.add_(h.sum())
        return h * 0.5

class _Stateful(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.register_buffer("ema", torch.zeros(4))
        self.register_buffer("steps", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("scale", torch.full((4,), 2.0))
        self.register_buffer("unused", torch.ones(3))
        self.register_buffer("scratch", torch.zeros(4), persistent=False)
        self.inner = _Inner()
    def forward(self, x):
        h = self.lin(x) * self.scale
        self.ema.mul_(0.9).add_(0.1 * h.mean(0))
        self.steps.add_(1)
        self.scratch.copy_(h.amax(0))
        return self.inner(h) + self.ema
""",
                @__MODULE__,
            )
            pydeepcopy = pyimport("copy").deepcopy
            buf(m, name) = torch_to_julia(m.get_buffer(name))

            for mode in (:eval, :train)
                @testset "$mode" begin
                    torch.manual_seed(0)
                    model = pyeval("_Stateful", @__MODULE__)()
                    mode === :train ? model.train() : model.eval()
                    ref = pydeepcopy(model)  # eager reference, mutated by its calls

                    tm, ps, st = Reactant.import_torch(model)
                    @test keys(st) ==
                        (:ema, :steps, :scale, :unused, :scratch, Symbol("inner.total"))

                    # Buffer updates in eval mode are flagged once per export; in
                    # train mode they are expected and not flagged.
                    x0 = Reactant.to_rarray(randn(Float32, 5, 4))
                    buffer_warnings() = filter(
                        l -> occursin("updates buffers", string(l.message)),
                        first(Test.collect_test_logs(() -> @jit(tm(x0, ps, st)))),
                    )
                    first_call = buffer_warnings()
                    if mode === :eval
                        @test length(first_call) == 1
                        @test occursin(
                            r"evaluation mode.*\"ema\".*\"steps\"",
                            string(only(first_call).message),
                        )
                    else
                        @test isempty(first_call)
                    end
                    @test isempty(buffer_warnings())  # the export is cached
                    @test st.steps isa ConcreteRNumber{Int64}
                    @test st.var"inner.total" isa ConcreteRNumber{Float32}
                    @test Reactant.torch_layout(tm, "inner.total").owner == "_Inner"
                    @test !Reactant.torch_layout(tm, "scratch").persistent
                    @test Reactant.torch_layout(tm, "ema").persistent
                    @test Reactant.torch_layout(tm, "steps").eltype === Int64
                    st0 = deepcopy(map(Array, (; st.ema, st.scale, st.unused)))

                    # Three steps, feeding the returned state into the next call.
                    st_k = st
                    for k in 1:3
                        xdata = randn(Float32, 5, 4)
                        y, st_k = @jit tm(Reactant.to_rarray(xdata), ps, st_k)
                        y_ref = ref(torch.from_numpy(np.asarray(xdata)))

                        @test relerr(y, torch_to_julia(y_ref)) < 1.0f-5
                        @test relerr(st_k.ema, buf(ref, "ema")) < 1.0f-5
                        @test Int(st_k.steps) == k
                        @test relerr(st_k.scratch, buf(ref, "scratch")) < 1.0f-5
                        @test abs(
                            Float32(st_k.var"inner.total") - only(buf(ref, "inner.total"))
                        ) <= 1.0f-4 * max(1, abs(only(buf(ref, "inner.total"))))
                        # Unwritten buffers pass through unchanged.
                        @test Array(st_k.scale) == st0.scale
                        @test Array(st_k.unused) == st0.unused
                        # Same structure and leaf types as the state passed in.
                        @test keys(st_k) == keys(st)
                        @test st_k.steps isa ConcreteRNumber{Int64}
                        @test st_k.var"inner.total" isa ConcreteRNumber{Float32}
                    end

                    # Functional semantics: neither the state passed in nor the
                    # torch module itself was modified.
                    @test Array(st.ema) == st0.ema
                    @test Int(st.steps) == 0
                    @test Float32(st.var"inner.total") == 0
                    @test all(iszero, buf(model, "ema"))
                    @test pyconvert(Int, model.steps.item()) == 0
                    @test pyconvert(Bool, model.training) == (mode === :train)

                    # The export returns exactly the written buffers, and the
                    # never-read buffer is not an argument.
                    exp = only(values(tm.cache))
                    mutated = Set(
                        s.target for s in exp.results if s.kind === :buffer_mutation
                    )
                    @test mutated == Set(["ema", "steps", "scratch", "inner.total"])
                    @test count(s -> s.kind === :user_output, exp.results) == 1
                    @test !any(s -> s.target == "unused", exp.arguments)

                    # A compiled step can be called repeatedly with the returned
                    # state, which requires st_new to have the same types as st.
                    step = @compile tm(Reactant.to_rarray(randn(Float32, 5, 4)), ps, st)
                    s = st
                    for _ in 1:3
                        _, s = step(Reactant.to_rarray(randn(Float32, 5, 4)), ps, s)
                    end
                    @test Int(s.steps) == 3
                end
            end
        end

        @testset "import_torch: display of a large model" begin
            # 100 Linear layers: 200 parameters. A limited display (as in the REPL)
            # abbreviates the table; an unlimited one lists every tensor.
            model = nn.Sequential([nn.Linear(2, 2) for _ in 1:100]...)
            tm, ps, st = Reactant.import_torch(model)
            @test length(Reactant.torch_layout(tm)) == 200

            limited = sprint(
                show,
                MIME"text/plain"(),
                tm;
                context=(:limit => true, :displaysize => (30, 120)),
            )
            @test occursin("parameters (ps), 200 tensors, 600 elements:", limited)
            @test occursin("buffers (st), 0 tensors, 0 elements:", limited)
            row(fqn) = Regex("^\\s+" * replace(fqn, "." => "\\.") * "\\s", "m")
            @test occursin(r"\d+ rows omitted", limited)
            @test occursin(row("0.weight"), limited)     # first rows kept
            @test occursin(row("99.bias"), limited)      # last rows kept
            @test !occursin(row("50.weight"), limited)   # middle elided
            @test count('\n', limited) < 30

            full = sprint(show, MIME"text/plain"(), tm)
            @test !occursin("omitted", full)
            @test all(i -> occursin(row("$(i).weight"), full), 0:99)
        end

        @testset "import_torch: parameter mutation is rejected" begin
            pyexec(
                """
import torch, torch.nn as nn
class _MutatesParam(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(3), requires_grad=False)
    def forward(self, x):
        self.w.add_(1.0)
        return x * self.w
""",
                @__MODULE__,
            )
            model = pyeval("_MutatesParam", @__MODULE__)()
            tm, ps, st = Reactant.import_torch(model)
            err = try
                @jit tm(Reactant.to_rarray(randn(Float32, 3)), ps, st)
                nothing
            catch e
                e
            end
            @test err !== nothing
            @test occursin("parameter", sprint(showerror, err))
        end

        @testset "import_torch: lifted constant ordering" begin
            # Same model as the shortcut test above: the lifted constant precedes
            # the BatchNorm buffers in the graph placeholders. The constant stays
            # embedded; ps and st are bound by FQN.
            torch.manual_seed(0)
            model = pyeval("_ConvBNConst", @__MODULE__)()
            model.eval()
            xdata = randn(Float32, 1, 3, 16, 16)
            y_native = model(torch.from_numpy(np.asarray(xdata)))
            tm, ps, st = Reactant.import_torch(model)
            @test unique(first.(split.(string.(keys(ps)), '.'))) == ["c1", "bn", "c2"]
            y, _ = @jit tm(Reactant.to_rarray(xdata), ps, st)
            @test relerr(y, torch_to_julia(y_native)) < 1.0f-4
        end

        @testset "import_torch: TorchScript" begin
            torch.manual_seed(0)
            conv = nn.Conv2d(3, 4, 3)
            conv.eval()
            scripted = torch.jit.trace(conv, torch.randn(1, 3, 8, 8))
            xdata = randn(Float32, 1, 3, 8, 8)
            y_native = scripted(torch.from_numpy(np.asarray(xdata)))
            tm, ps, st = Reactant.import_torch(scripted)
            @test keys(ps) == (:weight, :bias)
            y, st_new = @jit tm(Reactant.to_rarray(xdata), ps, st)
            @test st_new == NamedTuple()
            @test relerr(y, torch_to_julia(y_native)) < 1.0f-4
        end

        @testset "import_torch: TorchScript keeps parameters and buffers apart" begin
            # A traced module saved to and loaded from a file, as deployed models are.
            torch.manual_seed(0)
            model = nn.Sequential(nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU())
            model.eval()
            xdata = randn(Float32, 4, 8)
            x_torch = torch.from_numpy(np.asarray(xdata))
            buf = pyimport("io").BytesIO()
            torch.jit.save(torch.jit.trace(model, x_torch), buf)
            buf.seek(0)
            loaded = torch.jit.load(buf)

            tm, ps, st = Reactant.import_torch(loaded)
            kinds(m) = Dict(
                vcat(
                    [pyconvert(String, n) => :parameter for (n, _) in m.named_parameters()],
                    [pyconvert(String, n) => :buffer for (n, _) in m.named_buffers()],
                ),
            )
            @test Dict(e.fqn => e.kind for e in Reactant.torch_layout(tm)) == kinds(model)
            @test collect(string.(keys(ps))) == ["0.weight", "0.bias", "1.weight", "1.bias"]
            y, _ = @jit tm(Reactant.to_rarray(xdata), ps, st)
            @test relerr(y, torch_to_julia(model(x_torch))) < 1.0f-4
        end

        @testset "import_torch: full_like with an explicit layout" begin
            # `zeros_like(mask, dtype=..., layout=..., device=...)` (which TorchScript
            # artifacts record explicitly, for example in nn.TransformerEncoder's
            # padding mask) lowers to aten.full_like with a `layout` keyword, which
            # torchax's own handler does not accept.
            pyexec(
                """
import torch, torch.nn as nn
class _FullLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)
    def forward(self, x, mask):
        z = torch.zeros_like(mask, dtype=torch.float32, layout=torch.strided,
                             device=torch.device("cpu"))
        return self.lin(x) + z + torch.full_like(x, 2.0)
""",
                @__MODULE__,
            )
            torch.manual_seed(0)
            model = pyeval("_FullLike", @__MODULE__)()
            model.eval()
            xdata = randn(Float32, 3, 4)
            mdata = rand(Bool, 3, 4)
            x_torch = torch.from_numpy(np.asarray(xdata))
            m_torch = torch.from_numpy(np.asarray(mdata))
            for mod in (model, torch.jit.trace(model, (x_torch, m_torch)))
                y_native = mod(x_torch, m_torch)
                tm, ps, st = Reactant.import_torch(mod)
                y, _ = @jit tm(Reactant.to_rarray(xdata), Reactant.to_rarray(mdata), ps, st)
                @test relerr(y, torch_to_julia(y_native)) < 1.0f-4
            end
        end

        @testset "import_torch: multiple inputs and outputs" begin
            torch.manual_seed(0)
            model = pyeval("_TwoInput", @__MODULE__)()
            model.eval()
            adata = randn(Float32, 3, 8)
            bdata = randn(Float32, 3, 4)
            y_native = model(
                torch.from_numpy(np.asarray(adata)), torch.from_numpy(np.asarray(bdata))
            )
            tm, ps, st = Reactant.import_torch(model)
            y, _ = @jit tm(Reactant.to_rarray(adata), Reactant.to_rarray(bdata), ps, st)
            @test relerr(y, torch_to_julia(y_native)) < 1.0f-4

            model = pyeval("_TwoOutput", @__MODULE__)()
            model.eval()
            native = model(torch.from_numpy(np.asarray(adata)))
            tm, ps, st = Reactant.import_torch(model)
            y, _ = @jit tm(Reactant.to_rarray(adata), ps, st)
            @test y isa Tuple && length(y) == 2
            @test relerr(y[1], torch_to_julia(native[0])) < 1.0f-4
            @test relerr(y[2], torch_to_julia(native[1])) < 1.0f-4
        end

        @testset "import_torch: input mutation is rejected" begin
            pyexec(
                """
import torch, torch.nn as nn
class _MutatesInput(nn.Module):
    def forward(self, x):
        x.add_(1.0)
        return x * 2
""",
                @__MODULE__,
            )
            model = pyeval("_MutatesInput", @__MODULE__)()
            tm, ps, st = Reactant.import_torch(model)
            x = Reactant.to_rarray(randn(Float32, 3))
            err = try
                @jit tm(x, ps, st)
                nothing
            catch e
                e
            end
            @test err !== nothing
            @test occursin("in place", sprint(showerror, err))
        end

        @testset "import_torch: export labels match the StableHLO signature" begin
            torch.manual_seed(0)
            model = nn.Sequential(nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU())
            model.eval()
            xdata = randn(Float32, 4, 8)
            tm, ps, st = Reactant.import_torch(model)

            export_for(mode) =
                ReactantPythonCallExt._cached_export(tm, :torchax, mode, (xdata,))
            exp = export_for(:eval)
            kinds = [a.kind for a in exp.arguments]
            @test kinds == [
                :user_input,
                :parameter,
                :parameter,
                :parameter,
                :parameter,
                :buffer,
                :buffer,
            ]
            @test exp.arguments[1].target == 1
            @test exp.arguments[1].shape == (4, 8)
            # num_batches_tracked is not read in evaluation mode and is pruned.
            @test Set(a.target for a in exp.arguments if a.kind === :buffer) ==
                Set(["1.running_mean", "1.running_var"])
            for a in exp.arguments
                a.kind === :user_input && continue
                leaf = (a.kind === :parameter ? ps : st)[Symbol(a.target)]
                @test size(leaf) == a.shape
                @test Reactant.unwrapped_eltype(leaf) == a.eltype
            end
            @test [r.kind for r in exp.results] == [:user_output]
            # The entry function takes exactly the listed arguments.
            sig = match(r"func\.func public @main\((.*?)\) ->"s, exp.text)
            @test sig !== nothing
            @test length(collect(eachmatch(r"%arg\d+:", sig[1]))) == length(exp.arguments)

            # The export is cached in tm and reused by a traced call.
            @test length(tm.cache) == 1
            y, _ = @jit tm(Reactant.to_rarray(xdata), ps, st)
            @test length(tm.cache) == 1
            @test export_for(:eval) === exp

            # Training mode returns the buffer updates; the module keeps its mode.
            exp_train = export_for(:train)
            @test Set(r.target for r in exp_train.results if r.kind === :buffer_mutation) ==
                Set(["1.running_mean", "1.running_var", "1.num_batches_tracked"])
            @test !pyconvert(Bool, model.training)
            @test length(tm.cache) == 2

            @test exp.backend === :torchax
            @test_throws ArgumentError Reactant.import_torch(model; backend=:nope)
        end

        @testset "adaptive_max_pool2d" begin
            pyexec(
                """
import torch, torch.nn as nn
class _AdaptiveMax(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.pool = nn.AdaptiveMaxPool2d(size, return_indices=True)
    def forward(self, x):
        return self.pool(x)
""",
                @__MODULE__,
            )
            xdata = randn(Float32, 2, 3, 7, 9)
            x_torch = torch.from_numpy(np.asarray(xdata))
            for size in ((1, 1), (3, 3), (2, 5), (7, 9))
                model = pyeval("_AdaptiveMax", @__MODULE__)(pytuple(size))
                y_native, i_native = model(x_torch)
                for mod in (model, torch.jit.trace(model, x_torch))
                    tm, ps, st = Reactant.import_torch(mod)
                    (y, idx), _ = @jit tm(Reactant.to_rarray(xdata), ps, st)
                    @test Array(y) == torch_to_julia(y_native)
                    @test Array(idx) == torch_to_julia(i_native)
                end
            end
        end

        @testset "TorchScript modules with a fixed mode" begin
            torch.manual_seed(0)
            xdata = randn(Float32, 4, 8)
            x_torch = torch.from_numpy(np.asarray(xdata))

            # A scripted module that lists `training` in __constants__ raises on any
            # assignment to it, including `.eval()`. Scripting needs the source on disk.
            dir = mktempdir()
            write(
                joinpath(dir, "_reactant_const_training.py"),
                """
import torch.nn as nn
class ConstTraining(nn.Module):
    __constants__ = ["training"]
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 4)
        self.training = False
    def forward(self, x):
        return self.lin(x)
""",
            )
            sys = pyimport("sys")
            sys.path.insert(0, dir)
            try
                mod = pyimport("_reactant_const_training")
                scripted = torch.jit.script(mod.ConstTraining())
                @test_throws PyException scripted.train()
                tm, ps, st = Reactant.import_torch(scripted)
                y, _ = @jit tm(Reactant.to_rarray(xdata), ps, st)
                @test relerr(y, torch_to_julia(scripted(x_torch))) < 1.0f-4
                err = try
                    ReactantPythonCallExt._cached_export(tm, :torchax, :train, (xdata,))
                    nothing
                catch e
                    e
                end
                @test err !== nothing
                @test occursin("TorchScript constant", sprint(showerror, err))
            finally
                sys.path.remove(dir)
            end

            # A frozen module has no `training` attribute; its graph is evaluation
            # mode, and its parameters are folded into the graph as constants.
            model = nn.Sequential(nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU())
            model.eval()
            frozen = torch.jit.freeze(torch.jit.trace(model, x_torch))
            @test !pyhasattr(frozen, "training")
            tm, ps, st = Reactant.import_torch(frozen)
            y, _ = @jit tm(Reactant.to_rarray(xdata), ps, st)
            @test relerr(y, torch_to_julia(frozen(x_torch))) < 1.0f-4
            err = try
                ReactantPythonCallExt._cached_export(tm, :torchax, :train, (xdata,))
                nothing
            catch e
                e
            end
            @test err !== nothing
            @test occursin("frozen TorchScript module", sprint(showerror, err))
        end
    end
end
