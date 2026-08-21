# Plugin.jl — PJRT plugin lifecycle callbacks

function _plugin_initialize(args::Ptr{CAPI.PJRT_Plugin_Initialize_Args})::Ptr{Cvoid}
    return C_NULL
end

function _plugin_attributes(args::Ptr{CAPI.PJRT_Plugin_Attributes_Args})::Ptr{Cvoid}
    Reactant.unsafe_store_field!(args, Csize_t(0), Val{:num_attributes}())
    return C_NULL
end
