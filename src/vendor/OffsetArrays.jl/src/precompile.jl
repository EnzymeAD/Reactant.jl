function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{typeof(Base.showarg),IOBuffer,OffsetArray{Int, 0, Array{Int, 0}},Bool})   # time: 0.037824474
    Base.precompile(Tuple{Type{IdOffsetRange{Int, Base.OneTo{Int}}},UnitRange{Int}})   # time: 0.009825722
    Base.precompile(Tuple{typeof(Base.inds2string),Tuple{IdOffsetRange{Int, Base.OneTo{Int}}, IdOffsetRange{Int, Base.OneTo{Int}}}})   # time: 0.0080779
    Base.precompile(Tuple{typeof(zeros),Tuple{IdOffsetRange{Int, Base.OneTo{Int}}}})   # time: 0.007713056
    Base.precompile(Tuple{typeof(ones),Tuple{IdOffsetRange{Int, Base.OneTo{Int}}}})   # time: 0.007713056
    Base.precompile(Tuple{typeof(trues),Tuple{UnitRange{Int}, UnitRange{Int}}})   # time: 0.005478372
    Base.precompile(Tuple{typeof(falses),Tuple{UnitRange{Int}, UnitRange{Int}}})   # time: 0.005478372
    Base.precompile(Tuple{typeof(firstindex),IdOffsetRange{Int, Base.OneTo{Int}}})   # time: 0.004100289
end
