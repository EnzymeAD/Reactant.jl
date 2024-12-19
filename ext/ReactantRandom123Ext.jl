module ReactantRandom123Ext

using Random123: Threefry4x, Threefry2x, Philox4x, Philox2x
using Reactant: TracedRandom

TracedRandom.rng_algorithm(::Threefry4x) = "THREE_FRY"
TracedRandom.rng_algorithm(::Threefry2x) = "THREE_FRY"
TracedRandom.rng_algorithm(::Philox4x) = "PHILOX"
TracedRandom.rng_algorithm(::Philox2x) = "PHILOX"

end
