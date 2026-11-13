"""
    Extension

A PJRT extension that a plugin implements: an optional set of C functions that XLA looks up by type in the extension
chain of the `PJRT_Api`. Pass its type to [`make_api`](@ref) to advertise it to XLA. Since XLA only uses an extension
when it is in the chain, a plugin must pass only the extensions whose interface it implements.

Each subtype `E` is a `mutable struct` whose first field is the C struct of the extension (e.g.
`CAPI.PJRT_MemoryDescriptions_Extension`), so that a pointer to an `E` is a pointer to its C struct. Julia-only fields
(e.g. the `Base.CFunction`s of the C struct) can follow it.

### Interface

- `extension_type`
- `E(plugin::AbstractPlugin, next::Ptr{CAPI.PJRT_Extension_Base})`: build the C struct, whose `PJRT_Extension_Base`
  has the type of `E` and `next` as the next extension of the chain
"""
abstract type Extension end

"""
    extension_type(::Type{E}) where {E<:Extension} -> CAPI.PJRT_Extension_Type

Return the type that XLA identifies extensions of type `E` by. Each extension type can be passed only once to
[`make_api`](@ref).
"""
function extension_type end

include("MemoryDescriptions.jl")
include("FFI.jl")
include("Layouts.jl")
