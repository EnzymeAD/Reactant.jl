module Serialize

# TODO: move these deps into an extension
# TODO: Deal with sharding/global devices

using JLD2
using Reactant: Reactant, MLIR

struct SerializedThunk{FTy,tag,ArgTypes,IsClosure}
    f::FTy
    body::Expr
end

# function JLD2.writeas(
#     ::Type{<:Reactant.Compiler.Thunk{FTy,tag,ArgTypes,IsClosure}}
# ) where {FTy,tag,ArgTypes,IsClosure}
#     return SerializedThunk{FTy,tag,ArgTypes,IsClosure}
# end

# function JLD2.wconvert(
#     ::Type{SerializedThunk{FTy,tag,ArgTypes,IsClosure}},
#     thunk::Reactant.Compiler.Thunk{FTy,tag,ArgTypes,IsClosure},
# ) where {FTy,tag,ArgTypes,IsClosure}
#     if thunk.mod === nothing
#         throw("To serialize a compiled thunk, ensure it is called with `serializable=true`")
#     end

#     return error("TODO")
# end

# function JLD2.rconvert(
#     ::Type{Reactant.Compiler.Thunk{FTy,tag,ArgTypes,IsClosure}},
#     serialized::SerializedThunk{FTy,tag,ArgTypes,IsClosure},
# ) where {FTy,tag,ArgTypes,IsClosure}
#     return error("TODO")
# end

function serialize(
    thunk::Reactant.Compiler.Thunk{FTy,tag,ArgTypes,IsClosure}
) where {FTy,tag,ArgTypes,IsClosure}
    if thunk.mod === nothing
        throw("To serialize a compiled thunk, ensure it is called with `serializable=true`")
    end

    serializable_thunk = SerializedThunk{FTy,tag,ArgTypes,IsClosure}(
        thunk.f, Reactant.Compiler.__thunk_body_cache[tag]
    )
end

function deserialize() end

end
