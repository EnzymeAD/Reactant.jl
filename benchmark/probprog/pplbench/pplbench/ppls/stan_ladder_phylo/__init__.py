"""Stan placement variants of the phylogenetic Poisson regression (section 2).

The figure-12 comparison of ``stan_ladder`` (GP Poisson regression) repeated on
the section-2 model. Each subpackage is a ``package`` override for a ``stan``
PPL entry (``"package": "pplbench.ppls.stan_ladder_phylo.<variant>"``). All
variants receive the same data (``N``, ``x``, ``y`` and the tree covariance
``A``) and state the same density; they differ only in where the work on ``A``
is placed:

  naive     cov = sigma_sq * A, L = cholesky_decompose(cov), f = L * eta, all
            in ``model`` (the statistician's direct translation)
  slicstan  the SlicStan tool's output for the naive program (placement
            verbatim; declaration-order and syntax fix-ups only). SlicStan
            finds no data-only statement to move to ``transformed data`` (A
            arrives as data and every statement on it involves sigma_sq) and
            places cov, L and f in ``transformed parameters``.
  expert    brms's generated code for ``(1 | gr(phylo, cov = A))``: the
            Cholesky factor of A is computed once from data (brms computes
            ``Lcov <- t(chol(A))`` in R and passes it as data; here it is
            ``transformed data``) and the effect is ``r = sd * (Lcov * z)``
            with ``sd = sqrt(sigma_sq)``. This is the rewrite SICM performs.

There is no separate "data-only work hoisted by hand" variant: the program has
no data-only statement, so that variant (and SlicStan's placement written with
model locals, the GP ladder's ``slicstan_local``) is the naive program itself.

Provenance: oopsla/evidence/real_data_rows_2026-09-29/slicstan_phylo/ (SlicStan
input .slic and raw output .stan).
"""
