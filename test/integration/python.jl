using Reactant
using Reactant: Ops

using Test

fn(x, y) = sin.(x) .+ cos.(y.x[1:2, :])

@testset "No CondaPkg.toml in Reactant" begin
    @test !isfile(joinpath(pkgdir(Reactant), "CondaPkg.toml"))
end

@testset "SavedModel Error Before PythonCall Loaded" begin
    @test !Reactant.Serialization.serialization_supported(Val(:SavedModel))
    x = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 2, 10))
    y = (; x=Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 4, 10)))
    compiled_fn = @compile serializable = true fn(x, y)
    @test_throws "PythonCall" Reactant.Serialization.export_as_tf_saved_model(
        compiled_fn, mktempdir(), v"1.8.5", [], Dict()
    )
end

# Jax on Github CI dislikes X86 macos
@static if !Sys.isapple() || Sys.ARCH != :x86_64
    using PythonCall
    using CondaPkg
    using NPZ

    # Verify that loading ReactantPythonCallExt does not eagerly import jax or tensorflow.
    @testset "Lazy Python Imports & Missing Package Errors" begin
        sys = pyimport("sys")
        @test !pyconvert(Bool, pycontains(sys.modules, "jax"))
        @test !pyconvert(Bool, pycontains(sys.modules, "tensorflow"))

        # Calling Python on non-traced values inside @jit must not import jax or tensorflow.
        py_add_scalar = pyfunc((a, b) -> a + b)
        untraced_fn(x) = x .+ pyconvert(Float32, py_add_scalar(1.0f0, 2.0f0))
        r_untraced = @jit untraced_fn(Reactant.to_rarray(Float32[1, 2]))
        @test Array(r_untraced) ≈ Float32[4, 5]
        @test !pyconvert(Bool, pycontains(sys.modules, "jax"))
        @test !pyconvert(Bool, pycontains(sys.modules, "tensorflow"))

        # Simulate missing jax / tensorflow via PEP 302 (None in sys.modules) to check
        # error messages and verify subsequent recovery once available.
        pysetitem(sys.modules, "jax", pybuiltins.None)
        pysetitem(sys.modules, "tensorflow", pybuiltins.None)
        try
            py_double = pyfunc(x -> x + x)
            @test_throws "CondaPkg.add_pip(\"jax\")" (@jit py_double(
                Reactant.to_rarray(Float32[1, 2, 3])
            ))

            @test !Reactant.Serialization.serialization_supported(Val(:SavedModel))
            x = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 2, 10))
            y = (;
                x=Reactant.to_rarray(
                    Reactant.TestUtils.construct_test_array(Float32, 4, 10)
                )
            )
            compiled_fn = @compile serializable = true fn(x, y)
            @test_throws "CondaPkg.add_pip(\"tensorflow\")" Reactant.Serialization.export_as_tf_saved_model(
                compiled_fn, mktempdir(), v"1.8.5", [], Dict()
            )
        finally
            pydelitem(sys.modules, "jax")
            pydelitem(sys.modules, "tensorflow")
        end
    end

    # On macOS, dyld performs process-wide weak symbol coalescing between Julia's libLLVM
    # and TensorFlow's embedded LLVM (libtensorflow_framework / libtensorflow_cc). When TensorFlow
    # compiles the model via its XLA CPU backend, weak symbol interposition routes TensorFlow's LLVM
    # calls into Julia's libLLVM. Mismatches in LLVM C++ struct layout cause a SIGSEGV.
    # Running TensorFlow model evaluation in a child Python subprocess isolates TensorFlow in a process
    # image without Julia's libLLVM loaded.
    function eval_tf_saved_model_in_subprocess(
        model_path::String, inputs::Vector{<:AbstractArray}
    )
        tmpdir = mktempdir()
        input_paths = String[]
        for (i, arr) in enumerate(inputs)
            path = joinpath(tmpdir, "in_$(i).npy")
            npzwrite(path, arr)
            push!(input_paths, path)
        end
        out_path = joinpath(tmpdir, "out.npy")

        input_loads = join(["np.load(r\"$(path)\")" for path in input_paths], ", ")
        py_script = """
import tensorflow as tf
import numpy as np

restored_model = tf.saved_model.load(r"$model_path")
inputs = [$input_loads]
args = [tf.constant(arr) for arr in inputs]
res = restored_model.f(*args)[0].numpy()
np.save(r"$out_path", res)
"""
        script_path = joinpath(tmpdir, "eval.py")
        write(script_path, py_script)

        run(`$(CondaPkg.which("python")) $script_path`)
        return npzread(out_path)
    end

    @testset "PythonCall" begin
        sys = pyimport("sys")
        @test !pyconvert(Bool, pycontains(sys.modules, "jax"))

        # First traced pycall lazily imports jax without importing tensorflow.
        py_double = pyfunc(x -> x + x)
        doubled = @jit py_double(Reactant.to_rarray(Float32[1, 2, 3]))
        @test Array(doubled) ≈ Float32[2, 4, 6]
        @test pyconvert(Bool, pycontains(sys.modules, "jax"))
        @test !pyconvert(Bool, pycontains(sys.modules, "tensorflow"))

        jax = pyimport("jax")
        jnp = pyimport("jax.numpy")

        result = @jit jax.numpy.sum(Reactant.to_rarray(Float32[1, 2, 3]))
        @test result isa ConcreteRNumber{Float32}
        @test result ≈ 6

        pyfn = pyfunc(
            function (tup, y, num, partial_eval)
                return jnp.sin(tup[0]) + jnp.cos(tup[1]) + jnp.sum(y) - num * partial_eval
            end;
            name="sum_and_sin_cos_jax",
        )

        tup = (
            Reactant.TestUtils.construct_test_array(Float32, 3, 4),
            Reactant.TestUtils.construct_test_array(Float32, 3, 4),
        )
        y = Reactant.TestUtils.construct_test_array(Float32, 2, 2)
        num = 4.0
        partial_eval = 0.5

        gt = sin.(tup[1]) .+ cos.(tup[2]) .+ sum(y) .- num .* partial_eval

        tup_ra = Reactant.to_rarray(tup)
        y_ra = Reactant.to_rarray(y)
        num_ra = ConcreteRNumber{Float32}(num)

        result = @jit pyfn(tup_ra, y_ra, num_ra, partial_eval)
        @test result ≈ gt atol = 1e-5
        @test result isa ConcreteRArray{Float32,2}
    end

    @testset "SavedModel Export" begin
        sys = pyimport("sys")
        @test !pyconvert(Bool, pycontains(sys.modules, "tensorflow"))

        x = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 2, 10))
        y = (;
            x=Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 4, 10))
        )

        compiled_fn = @compile serializable = true fn(x, y)

        true_res = Array(compiled_fn(x, y))

        @testset "Serialize without parameters" begin
            model_path = tempdir() * "/test_saved_model_1"
            Reactant.Serialization.export_as_tf_saved_model(
                compiled_fn, model_path, v"1.8.5", [], Dict()
            )
            @test pyconvert(Bool, pycontains(sys.modules, "tensorflow"))
            @test Reactant.Serialization.serialization_supported(Val(:SavedModel))

            if Sys.isapple()
                res_raw = eval_tf_saved_model_in_subprocess(
                    model_path,
                    [permutedims(Array(x), (2, 1)), permutedims(Array(y.x), (2, 1))],
                )
                res = permutedims(res_raw, (2, 1))
            else
                tf = pyimport("tensorflow")
                np = pyimport("numpy")
                restored_model = tf.saved_model.load(model_path)
                res = permutedims(
                    PyArray(
                        np.asarray(
                            restored_model.f(
                                tf.constant(np.asarray(permutedims(Array(x), (2, 1)))),
                                tf.constant(np.asarray(permutedims(Array(y.x), (2, 1)))),
                            )[0],
                        ),
                    ),
                    (2, 1),
                )
            end
            @test res ≈ true_res
        end

        @testset "Serialize with parameters" begin
            model_path = tempdir() * "/test_saved_model_2"
            Reactant.Serialization.export_as_tf_saved_model(
                compiled_fn, model_path, v"1.8.5", [1, "y.x"], Dict("y.x" => y.x)
            )

            if Sys.isapple()
                res_raw = eval_tf_saved_model_in_subprocess(
                    model_path, [permutedims(Array(x), (2, 1))]
                )
                res = permutedims(res_raw, (2, 1))
            else
                tf = pyimport("tensorflow")
                np = pyimport("numpy")
                restored_model = tf.saved_model.load(model_path)
                res = permutedims(
                    PyArray(
                        np.asarray(
                            restored_model.f(
                                tf.constant(np.asarray(permutedims(Array(x), (2, 1))))
                            )[0],
                        ),
                    ),
                    (2, 1),
                )
            end
            @test res ≈ true_res
        end
    end
end
