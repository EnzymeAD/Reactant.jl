# Phylogenetic Poisson regression data (motivating example)

Source: the worked example of the brms vignette "Estimating Phylogenetic Multilevel
Models with brms" (P.-C. Buerkner), whose data come from chapter 11 of
*Modern Phylogenetic Comparative Methods and Their Application in Evolutionary Biology*
(de Villemereuil & Nakagawa, 2014; https://www.mpcm-evolution.com/).

Files (downloaded 2026-08-22, unmodified):
- `phylo.nex`     https://paul-buerkner.github.io/data/phylo.nex      (200-species tree, NEXUS; written by ape)
- `data_pois.txt` https://paul-buerkner.github.io/data/data_pois.txt  (columns: phen_pois, cofactor, phylo)

The brms model on these data is `phen_pois ~ cofactor + (1|gr(phylo, cov = A)) + (1|obs)`,
`family = poisson("log")`, with `A <- ape::vcv.phylo(phylo)`.
`pplbench/pplbench/models/phylo_pois.py` reproduces `vcv.phylo` (shared root-to-MRCA path
length) from the NEXUS file directly; the tree is ultrametric with height 1, so `A` has unit
diagonal.
