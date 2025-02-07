module ReactantRandom123Ext

using Random123: Threefry4x, Threefry2x, Philox4x, Philox2x
using Reactant: TracedRandom
using Reactant.MLIR.Dialects: stablehlo

TracedRandom.rng_algorithm(::Threefry4x) = stablehlo.RngAlgorithm.THREE_FRY
TracedRandom.rng_algorithm(::Threefry2x) = stablehlo.RngAlgorithm.THREE_FRY
TracedRandom.rng_algorithm(::Philox4x) = stablehlo.RngAlgorithm.PHILOX
TracedRandom.rng_algorithm(::Philox2x) = stablehlo.RngAlgorithm.PHILOX

end
