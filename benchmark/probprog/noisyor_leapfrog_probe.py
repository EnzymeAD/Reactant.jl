import sys, time, json
import numpy as np
sys.path.insert(0, "/mnt4/sbrantq/probprog/oopsla/Reactant.jl/benchmark/probprog/pplbench")
from pplbench.models.noisy_or_topic import NoisyOrTopic as Model

ARGS = dict(n=2, num_topics=50, num_words=100, avg_fanout=3.0, avg_leak_weight=0.1, avg_topic_weight=1.0)
train, test = Model.generate_data(seed=12345, **ARGS)
edge_weight = train.attrs["edge_weight"] if "edge_weight" in train.attrs else None
print("train dims:", dict(train.sizes), "attrs:", list(train.attrs)[:5])

which = sys.argv[1]
NW, NS = 1000, 4000

if which == "numpyro":
    from pplbench.ppls.numpyro.noisy_or_topic import NoisyOrTopic as Impl
    import jax, numpyro
    from numpyro.infer import MCMC, NUTS
    attrs = dict(ARGS); attrs["n"] = 1
    attrs["edge_weight"] = train.attrs["edge_weight"]
    impl = Impl(**attrs)
    kernel = NUTS(impl.model, step_size=0.1, max_tree_depth=10,
                  adapt_step_size=True, adapt_mass_matrix=True, target_accept_prob=0.8)
    mcmc = MCMC(kernel, num_warmup=NW, num_samples=NS, progress_bar=False)
    key = jax.random.PRNGKey(0)
    t0 = time.perf_counter()
    mcmc.run(key, train, extra_fields=("num_steps",))
    wall = time.perf_counter() - t0
    ns = np.asarray(mcmc.get_extra_fields()["num_steps"])
    total_lf = int(ns.sum())
    print(f"NUMPYRO wall={wall:.1f}s draws={NS} leapfrogs(post-warmup)={total_lf} "
          f"avg_lf/draw={ns.mean():.1f} per-leapfrog={wall/ (total_lf * (NW+NS)/NS) *1e6:.1f}us(approx incl warmup scaling)")
    # cleaner: assume warmup lf/draw similar; per-lf = wall / (ns.mean() * (NW+NS))
    print(f"NUMPYRO per-leapfrog ~= {wall / (ns.mean() * (NW + NS)) * 1e6:.1f} us")
elif which == "stan":
    from pplbench.ppls.stan.noisy_or_topic import NoisyOrTopic as SImpl
    import cmdstanpy, tempfile, os
    attrs = dict(ARGS); attrs["n"] = 1
    attrs["edge_weight"] = train.attrs["edge_weight"]
    simpl = SImpl(**attrs)
    sdata = simpl.format_data_to_stan(train)
    sdata = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in sdata.items()}
    sdata["n"] = 1
    code = simpl.get_code()
    d = tempfile.mkdtemp()
    fn = os.path.join(d, "noisyor.stan")
    open(fn, "w").write(code)
    m = cmdstanpy.CmdStanModel(stan_file=fn)
    t0 = time.perf_counter()
    fit = m.sample(data=sdata, chains=1, iter_warmup=NW, iter_sampling=NS,
                   step_size=0.1, max_treedepth=10, adapt_engaged=True, seed=0, show_progress=False)
    wall = time.perf_counter() - t0
    lf = fit.method_variables()["n_leapfrog__"].astype(int)
    total_lf = int(lf.sum())
    print(f"STAN wall={wall:.1f}s draws={NS} leapfrogs(post-warmup)={total_lf} avg_lf/draw={lf.mean():.1f}")
    print(f"STAN per-leapfrog ~= {wall / (lf.mean() * (NW + NS)) * 1e6:.1f} us")
