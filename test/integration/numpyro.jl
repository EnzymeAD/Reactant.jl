using Reactant, Test, PythonCall

const jnp = pyimport("jax.numpy")
const jax = pyimport("jax")
const numpyro = pyimport("numpyro")

# Based on https://num.pyro.ai/en/stable/examples/hmcecs.html#example-hamiltonian-monte-carlo-with-energy-conserving-subsampling
@testset "NumPyro: HMC" begin
    model = pyfunc(
        function (data, obs)
            n, m = data.shape
            theta = numpyro.sample(
                "theta", numpyro.distributions.Normal(jnp.zeros(m), 0.5 * jnp.ones(m))
            )
            return pywith(numpyro.plate("N", n)) do _
                logits = jnp.matmul(data, theta)
                return numpyro.sample(
                    "obs", numpyro.distributions.Bernoulli(; logits=logits); obs=obs
                )
            end
        end;
        name="model",
    )

    run_hmc = pyfunc(
        function (mcmc_key, data, obs)
            kernel = numpyro.infer.HMC(model)
            mcmc = numpyro.infer.MCMC(kernel; num_warmup=1, num_samples=16)
            mcmc.run(mcmc_key, data, obs)
            return mcmc.get_samples()
        end; name="run_hmc"
    )

    data = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 10, 28))
    obs = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(Float32, 10))

    hmc_key = Reactant.to_rarray(Reactant.TestUtils.construct_test_array(UInt32, 1, 2))

    result = @jit run_hmc(hmc_key, data, obs)
    @test result isa ConcreteRArray
    @test size(result) == (16, 1, 28)
end
