"""Stan optimization-ladder variants of the FUSED heat-inverse + GP-prior model.

Sibling of ``stan_ladder`` (the GP Poisson regression ladder). A separate package
so that nothing belonging to the existing GP ladder is touched.

Each subpackage is a drop-in ``package`` override for a ``stan`` PPL entry in a
pplbench config (``"package": "pplbench.ppls.stan_ladder_fused.<variant>"``). All
variants share the data layout of ``_fused_ladder_base`` (raw ``s``, ``Lap``,
``H``, ``y`` -- no precomputed kernel and no precomputed factorization) and are
expression-for-expression identical apart from the placement being studied:

  naive           everything in ``model`` (the statistician's direct translation)
  slicstan        the SlicStan tool's output (placement verbatim; fix-ups only)
  slicstan_subst  same placement, with SlicStan's hand-written forward/back
                  substitution instead of Stan's library triangular solve
                  (SlicStan's grammar has no mdivide_*; this rung measures what
                  that spelling costs, so the other rungs can use the library
                  solve uniformly)
  expert_a        TIER 1 by hand: chol(M(kappa)) hoisted out of the time loop
  expert_b        expert_a + TIER 2a by hand (CholScale): chol(K) in
                  ``transformed data``, u0 = tau * (L_K * u_tilde)
  expert_c        expert_b + TIER 2b by hand (EigenLift): eigendecomposition of
                  Lap in ``transformed data``, no factorization per gradient

Provenance: oopsla/related_work/slic/heat_gp_fused_naive.slic (SlicStan input)
and heat_gp_fused_slicstan_raw.stan (verbatim tool output); analysis in
oopsla/evidence/related_work/fused_ladder_pilot_2026-08-25.md.
"""
