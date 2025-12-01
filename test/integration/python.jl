using Reactant
using Reactant: Ops

using Test

fn(x, y) = sin.(x) .+ cos.(y.x[1:2, :])

# Jax on Github CI dislikes X86 macos
@static if !Sys.isapple() || Sys.ARCH != :x86_64
    using PythonCall

    @testset "PythonCall" begin
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
        tf = pyimport("tensorflow")
        np = pyimport("numpy")

        x = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 2, 10))
        y = (;
            x=Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 4, 10))
        )

        compiled_fn = @compile serializable = true fn(x, y)

        true_res = Array(compiled_fn(x, y))

        @testset "Serialize without parameters" begin
            Reactant.Serialization.export_as_tf_saved_model(
                compiled_fn, tempdir() * "/test_saved_model_1", v"1.8.5", [], Dict()
            )

            restored_model = tf.saved_model.load(tempdir() * "/test_saved_model_1")

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
            @test res ≈ true_res
        end

        @testset "Serialize with parameters" begin
            Reactant.Serialization.export_as_tf_saved_model(
                compiled_fn,
                tempdir() * "/test_saved_model_2",
                v"1.8.5",
                [1, "y.x"],
                Dict("y.x" => y.x),
            )

            restored_model = tf.saved_model.load(tempdir() * "/test_saved_model_2")

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
            @test res ≈ true_res
        end
    end
end
