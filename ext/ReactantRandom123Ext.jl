module ReactantRandom123Ext

using Random123: Threefry4x, Threefry2x, Philox4x, Philox2x
using Reactant: TracedRandom

TracedRandom.rng_algorithm(::Threefry4x) = "THREE_FRY"
TracedRandom.rng_algorithm(::Philox4x) = "PHILOX"

TracedRandom.rng_algorithm(::Threefry2x) = "THREE_FRY"
TracedRandom.should_warn_if_not_natively_supported(::Threefry2x) = nothing
TracedRandom.make_seed(rng::Threefry2x) = UInt64[rng.key1, rng.key2]

TracedRandom.rng_algorithm(::Philox2x) = "PHILOX"
TracedRandom.should_warn_if_not_natively_supported(::Philox2x) = nothing
TracedRandom.make_seed(rng::Philox2x) = UInt64[rng.ctr1, rng.ctr2, rng.key]

end
