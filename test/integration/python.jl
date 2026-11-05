using Reactant
using Reactant: Ops
using BFloat16s: BFloat16

using Test

fn(x, y) = sin.(x) .+ cos.(y.x[1:2, :])

function multi_fn(x32, x64, flag, w1, b1, cparam, bfparam)
    mat = (w1 * x32) .+ b1 .+ bfparam
    gated = ifelse.(flag, mat, -mat)
    scalar = sum(gated) + sum(x64)
    cout = gated .+ cparam
    return gated, scalar, cout
end

@testset "Pure-Julia SavedModel Export" begin
    @test !occursin("tensorflow", read(joinpath(pkgdir(Reactant), "CondaPkg.toml"), String))
    @test Reactant.Serialization.serialization_supported(Val(:SavedModel))

    x = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 2, 10))
    y = (; x=Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 4, 10)))

    non_serializable = @compile fn(x, y)
    @test_throws ErrorException Reactant.Serialization.export_as_tf_saved_model(
        non_serializable, joinpath(mktempdir(), "bad_sm"), v"1.8.5"
    )

    compiled_fn = @compile serializable = true fn(x, y)

    dir_no_params = joinpath(mktempdir(), "pure_sm_no_params")
    Reactant.Serialization.export_as_tf_saved_model(
        compiled_fn, dir_no_params, v"1.8.5", [], Dict()
    )
    @test isfile(joinpath(dir_no_params, "saved_model.pb"))
    @test filesize(joinpath(dir_no_params, "saved_model.pb")) > 0
    @test isfile(joinpath(dir_no_params, "variables", "variables.index"))
    @test isfile(joinpath(dir_no_params, "variables", "variables.data-00000-of-00001"))

    dir_with_params = joinpath(mktempdir(), "pure_sm_with_params")
    Reactant.Serialization.export_as_tf_saved_model(
        compiled_fn, dir_with_params, v"1.8.5", [1, "y.x"], Dict("y.x" => y.x)
    )
    @test isfile(joinpath(dir_with_params, "saved_model.pb"))
    @test filesize(
        joinpath(dir_with_params, "variables", "variables.data-00000-of-00001")
    ) > 0

    # Validation error checks
    @test_throws ArgumentError Reactant.Serialization.export_as_tf_saved_model(
        compiled_fn, joinpath(mktempdir(), "missing_key"), v"1.8.5", [1, "y.x"], Dict()
    )
    @test_throws ArgumentError Reactant.Serialization.export_as_tf_saved_model(
        compiled_fn,
        joinpath(mktempdir(), "bad_shape"),
        v"1.8.5",
        [1, "y.x"],
        Dict("y.x" => zeros(Float32, 2, 10)),
    )
    @test_throws ArgumentError Reactant.Serialization.export_as_tf_saved_model(
        compiled_fn,
        joinpath(mktempdir(), "bad_dtype"),
        v"1.8.5",
        [1, "y.x"],
        Dict("y.x" => zeros(Float64, 4, 10)),
    )
    @test_throws ArgumentError Reactant.Serialization.export_as_tf_saved_model(
        compiled_fn,
        joinpath(mktempdir(), "bad_pos"),
        v"1.8.5",
        [2, "y.x"],
        Dict("y.x" => y.x),
    )
end

# Jax on Github CI dislikes X86 macos
@static if !Sys.isapple() || Sys.ARCH != :x86_64
    using PythonCall
    using CondaPkg
    using NPZ

    function active_python_executable()
        exe = PythonCall.python_executable_path()
        !isempty(exe) && return exe
        return CondaPkg.which("python")
    end

    # On macOS, dyld performs process-wide weak symbol coalescing between Julia's libLLVM
    # and TensorFlow's embedded LLVM (libtensorflow_framework / libtensorflow_cc). When TensorFlow
    # compiles the model via its XLA CPU backend, weak symbol interposition routes TensorFlow's LLVM
    # calls into Julia's libLLVM. Mismatches in LLVM C++ struct layout cause a SIGSEGV.
    # Running TensorFlow model evaluation in a child Python subprocess isolates TensorFlow in a process
    # image without Julia's libLLVM loaded.
    function eval_tf_saved_model_in_subprocess(
        model_path::String,
        inputs::Vector{<:AbstractArray};
        check_resave::Bool=false,
        check_v1_loader::Bool=true,
    )
        tmpdir = mktempdir()
        input_paths = String[]
        for (i, arr) in enumerate(inputs)
            path = joinpath(tmpdir, "in_$(i).npy")
            npzwrite(path, arr)
            push!(input_paths, path)
        end

        input_loads = join(["np.load(r\"$(path)\")" for path in input_paths], ", ")
        py_script = """
import os
import tensorflow as tf
import numpy as np

model_path = r"$model_path"
tmpdir = r"$tmpdir"
inputs = [$input_loads]
args = [tf.constant(arr) for arr in inputs]
kwargs = {f"args_{i}": arg for i, arg in enumerate(args)}

restored_model = tf.saved_model.load(model_path)
f_outs = [t.numpy() for t in restored_model.f(*args)]
sig_dict = restored_model.signatures["serving_default"](**kwargs)
sig_outs = [sig_dict[f"output_{i}"].numpy() for i in range(len(f_outs))]
for a, b in zip(f_outs, sig_outs):
    np.testing.assert_allclose(a, b)

if $(check_resave ? "True" : "False"):
    resaved_path = os.path.join(tmpdir, "resaved")
    tf.saved_model.save(restored_model, resaved_path)
    reloaded = tf.saved_model.load(resaved_path)
    for a, b in zip(f_outs, [t.numpy() for t in reloaded.f(*args)]):
        np.testing.assert_allclose(a, b)

if $(check_v1_loader ? "True" : "False"):
    with tf.compat.v1.Graph().as_default(), tf.compat.v1.Session() as sess:
        meta = tf.compat.v1.saved_model.loader.load(sess, ["serve"], model_path)
        sig_def = meta.signature_def["serving_default"]
        feed = {
            sess.graph.get_tensor_by_name(sig_def.inputs[f"args_{i}"].name): arr
            for i, arr in enumerate(inputs)
        }
        fetches = [
            sess.graph.get_tensor_by_name(sig_def.outputs[f"output_{i}"].name)
            for i in range(len(f_outs))
        ]
        v1_outs = sess.run(fetches, feed_dict=feed)
        for a, b in zip(f_outs, v1_outs):
            np.testing.assert_allclose(a, b)

for i, out in enumerate(f_outs):
    np.save(os.path.join(tmpdir, f"out_{i}.npy"), out)
with open(os.path.join(tmpdir, "num_outs.txt"), "w") as f:
    f.write(str(len(f_outs)))
"""
        script_path = joinpath(tmpdir, "eval.py")
        write(script_path, py_script)

        run(`$(active_python_executable()) $script_path`)
        num_outs = parse(Int, strip(read(joinpath(tmpdir, "num_outs.txt"), String)))
        return [npzread(joinpath(tmpdir, "out_$(i - 1).npy")) for i in 1:num_outs]
    end

    @testset "PythonCall" begin
        @test pyis(pyimport("sys").modules.get("tensorflow"), pybuiltins.None)
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
        @test pyis(pyimport("sys").modules.get("tensorflow"), pybuiltins.None)
    end

    @testset "SavedModel Export" begin
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

            if Sys.isapple()
                outs = eval_tf_saved_model_in_subprocess(
                    model_path,
                    [permutedims(Array(x), (2, 1)), permutedims(Array(y.x), (2, 1))],
                )
                res = permutedims(outs[1], (2, 1))
            else
                tf = pyimport("tensorflow")
                np = pyimport("numpy")
                restored_model = tf.saved_model.load(model_path)
                tf_x = tf.constant(np.asarray(permutedims(Array(x), (2, 1))))
                tf_y = tf.constant(np.asarray(permutedims(Array(y.x), (2, 1))))
                res = permutedims(
                    PyArray(np.asarray(restored_model.f(tf_x, tf_y)[0])), (2, 1)
                )
                sig_out = restored_model.signatures["serving_default"](;
                    args_0=tf_x, args_1=tf_y
                )["output_0"]
                @test permutedims(PyArray(np.asarray(sig_out)), (2, 1)) ≈ true_res
            end
            @test res ≈ true_res
        end

        @testset "Serialize with parameters" begin
            model_path = tempdir() * "/test_saved_model_2"
            Reactant.Serialization.export_as_tf_saved_model(
                compiled_fn, model_path, v"1.8.5", [1, "y.x"], Dict("y.x" => y.x)
            )

            outs = eval_tf_saved_model_in_subprocess(
                model_path,
                [permutedims(Array(x), (2, 1))];
                check_resave=true,
                check_v1_loader=true,
            )
            res = permutedims(outs[1], (2, 1))
            @test res ≈ true_res

            if !Sys.isapple()
                tf = pyimport("tensorflow")
                np = pyimport("numpy")
                restored_model = tf.saved_model.load(model_path)
                tf_x = tf.constant(np.asarray(permutedims(Array(x), (2, 1))))
                res_inproc = permutedims(
                    PyArray(np.asarray(restored_model.f(tf_x)[0])), (2, 1)
                )
                @test res_inproc ≈ true_res
            end
        end

        @testset "Multiple parameters, outputs, scalars, and dtypes" begin
            x32_jl = Float32[1 2 3 4 5; 6 7 8 9 10; 11 12 13 14 15]
            x64_jl = Float64[0.5, -1.5, 2.0, 3.25, -0.75]
            w1_jl = Float32[0.25 -0.5 1.0; 1.5 0.5 -0.25]
            b1_jl = Int32[1 -2 3 -4 5; 6 -7 8 -9 10]
            c_jl = ComplexF32[
                1+2im 3-1im 0.5+0im -2+4im 1-1im
                -1+0.5im 2+2im -3+1im 0-2im 4+0.25im
            ]
            bf_jl = BFloat16[
                0.5 -1.0 1.5 -2.0 0.25
                1.0 0.5 -0.5 2.0 -1.5
            ]

            x32 = Reactant.to_rarray(x32_jl)
            x64 = Reactant.to_rarray(x64_jl)
            flag_true = ConcreteRNumber(true)
            w1 = Reactant.to_rarray(w1_jl)
            b1 = Reactant.to_rarray(b1_jl)
            cparam = Reactant.to_rarray(c_jl)
            bfparam = Reactant.to_rarray(bf_jl)

            compiled_multi = @compile serializable = true multi_fn(
                x32, x64, flag_true, w1, b1, cparam, bfparam
            )
            gt_gated, gt_scalar, gt_cout = compiled_multi(
                x32, x64, flag_true, w1, b1, cparam, bfparam
            )
            gt_gated_neg, gt_scalar_neg, gt_cout_neg = compiled_multi(
                x32, x64, ConcreteRNumber(false), w1, b1, cparam, bfparam
            )

            model_path = joinpath(mktempdir(), "multi_sm")
            Reactant.Serialization.export_as_tf_saved_model(
                compiled_multi,
                model_path,
                v"1.8.5",
                [1, 2, 3, "layer.w", "_bias[0]", "complex_w", "bf16_w"],
                Dict(
                    "layer.w" => w1,
                    "_bias[0]" => b1,
                    "complex_w" => cparam,
                    "bf16_w" => bfparam,
                ),
            )

            outs_true = eval_tf_saved_model_in_subprocess(
                model_path,
                AbstractArray[permutedims(x32_jl, (2, 1)), x64_jl, fill(true)];
                check_resave=true,
                check_v1_loader=true,
            )
            @test permutedims(outs_true[1], (2, 1)) ≈ Array(gt_gated) atol = 1e-4
            @test only(outs_true[2]) ≈ Reactant.to_number(gt_scalar) atol = 1e-4
            @test permutedims(outs_true[3], (2, 1)) ≈ Array(gt_cout) atol = 1e-4

            outs_false = eval_tf_saved_model_in_subprocess(
                model_path,
                AbstractArray[permutedims(x32_jl, (2, 1)), x64_jl, fill(false)];
                check_resave=false,
                check_v1_loader=true,
            )
            @test permutedims(outs_false[1], (2, 1)) ≈ Array(gt_gated_neg) atol = 1e-4
            @test only(outs_false[2]) ≈ Reactant.to_number(gt_scalar_neg) atol = 1e-4
            @test permutedims(outs_false[3], (2, 1)) ≈ Array(gt_cout_neg) atol = 1e-4
        end
    end
end
