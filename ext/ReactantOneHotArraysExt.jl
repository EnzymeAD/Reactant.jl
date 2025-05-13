module ReactantOffsetArraysExt

using OneHotArrays
using Reactant

function Reactant.traced_type_inner(
    @nospecialize(_::Type{OneHotArrays.OneHotArray{T, N, Np1, I}}),
    seen,
    @nospecialize(mode::Reactant.TraceMode),
    @nospecialize(track_numbers::Type),
    @nospecialize(sharding),
    @nospecialize(runtime)
) where {T, N, Np1, I}
	I2 = Reactant.traced_type_inner(I, seen, mode, track_numbers, sharding, runtime)
	T2 = if eltype(I2) <: Reactant.TracedRNumber && !(T <: Reactant.TracedRNumber)
		Reactant.TracedRNumber{T}
	else
		T
	end
	@show I2, T2
	@show eltype(I2), eltype(I2) <: Reactant.TracedRNumber
    return OneHotArrays.OneHotArray{T2, N, Np1, I2}
end

end
