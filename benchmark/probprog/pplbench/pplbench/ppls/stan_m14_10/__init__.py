"""Stan programs of m14.10 for the section-2 Stan comparison.

Each subpackage is a ``package`` override for a ``stan`` PPL entry
(``"package": "pplbench.ppls.stan_m14_10.<variant>"``). Both variants receive the
same data, the data list ``rethinking::ulam`` builds for R code 14.51 (``N_spp``,
``B``, ``G``, ``M``, ``R`` and the unused ``V``, ``Imat``), declare the same
parameters in the same order, and state the same density. They differ only in
where the work on ``R`` is placed:

  textbook  the program ``rethinking::ulam(..., sample = FALSE)`` generates for
            the formula of R code 14.51 (rethinking ac1b3b2), verbatim:
            ``B ~ multi_normal(mu, R * sigma_sq)`` in ``model``, so Stan factors
            the covariance at every gradient evaluation.
  expert    the rewrite of the Stan user guide (Gaussian-process chapter): the
            Cholesky factor of the fixed matrix is computed once in
            ``transformed data`` and the likelihood is
            ``multi_normal_cholesky(mu, sqrt(sigma_sq) * L_R)``. Every other
            statement is the textbook program unchanged.

Provenance: oopsla/evidence/section2_m14_10_reference_2026-10-07/ (ulam code
generation, data list md5 ecc3b546) and section2_stan_timing_2026-10-12/.
"""
