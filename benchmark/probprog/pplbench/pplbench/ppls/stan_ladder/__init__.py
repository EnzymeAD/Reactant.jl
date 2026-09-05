"""Stan optimization-ladder variants of the GP Poisson regression model.

Each subpackage is a drop-in ``package`` override for a ``stan`` PPL entry in a
pplbench config (``"package": "pplbench.ppls.stan_ladder.<variant>"``). All
variants share the data layout of ``_gp_ladder_base`` (raw ``x``, not a
precomputed kernel) and differ ONLY in where the kernel construction and the
Cholesky factorization are placed:

  naive           everything in ``model`` (the statistician's direct translation)
  slicstan        the SlicStan tool's output (placement verbatim; syntax fix-ups only)
  slicstan_local  SlicStan's placement with the parameter-level statements as
                  ``model`` locals instead of ``transformed parameters`` (removes
                  Stan's per-draw CSV output of L and cov; placement unchanged)
  expert          the CholScale rewrite applied by hand: chol(K) in
                  ``transformed data``, f = alpha * (L_K * f_tilde)

Provenance: evidence/related_work/ (SlicStan input .slic + raw output .stan).
"""
