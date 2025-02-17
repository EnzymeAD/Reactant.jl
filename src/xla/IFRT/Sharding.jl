# mutable struct HloSharding
#     ptr::Ptr{Cvoid}

#     function HloSharding(ptr::Ptr{Cvoid})
#         @assert ptr != C_NULL
#         return finalizer(free_hlo_module, new(ptr))
#     end
# end

# function free_hlo_module(hlo_module)
#     @ccall MLIR.API.mlir_c.free_ifrt_hlo_module(hlo_module.ptr::Ptr{Cvoid})::Cvoid
# end

# function Base.convert(::Type{HloSharding}, xla_hlo_module::XLA.HloModule)
#     hlo_module = Ref{Ptr{Cvoid}}()
#     GC.@preserve xla_hlo_module hlo_module begin
#         @ccall MLIR.API.mlir_c.ifrt_hlo_module_from_xla_hlo_module(
#             xla_hlo_module.ptr::Ptr{Cvoid}, hlo_module::Ptr{Ptr{Cvoid}}
#         )
#     end
#     return HloSharding(hlo_module[])
# end
