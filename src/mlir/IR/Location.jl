@checked struct Location
    ref::API.MlirLocation
    # the context the location lives in, and its generation, to tell whether
    # it is still alive (see `live_generation`)
    context::API.MlirContext
    generation::UInt
end

function Location(ref::API.MlirLocation)
    ctx = API.mlirLocationGetContext(ref)
    return Location(ref, ctx, live_generation(ctx))
end

# Whether the location's context has not been disposed of (a context not made
# by `Context` is taken as alive).
function isalive(location::Location)
    return location.generation == 0 ||
           live_generation(location.context) == location.generation
end

function Location(; context::Context=current_context())
    return Location(API.mlirLocationUnknownGet(context))
end

function Location(filename, line, column; context::Context=current_context())
    return Location(API.mlirLocationFileLineColGet(context, filename, line, column))
end

function Location(callee::Location, caller::Location; context::Context=current_context())
    return Location(API.mlirLocationCallSiteGet(callee, caller))
end

function Location(name::String, location::Location; context::Context=current_context())
    return Location(API.mlirLocationNameGet(context, name, location))
end

Base.cconvert(::Core.Type{API.MlirLocation}, location::Location) = location
Base.unsafe_convert(::Core.Type{API.MlirLocation}, location::Location) = location.ref
# C takes an array of the bare references (`Location` carries more than its reference)
function Base.cconvert(::Core.Type{Ptr{API.MlirLocation}}, locations::Vector{Location})
    return API.MlirLocation[location.ref for location in locations]
end

Base.:(==)(a::Location, b::Location) = API.mlirLocationEqual(a, b)
context(location::Location) = Context(API.mlirLocationGetContext(location))

function Base.show(io::IO, location::Location)
    isalive(location) || return print(io, "Location(#= of a disposed context =#)")
    c_print_callback = @cfunction(print_callback, Cvoid, (API.MlirStringRef, Any))
    ref = Ref(io)
    print(io, "Location(#= ")
    API.mlirLocationPrint(location, c_print_callback, ref)
    return print(io, " =#)")
end

# TODO(#2245): rename to merge?
function fuse(locations::Vector{Location}, metadata; context::Context=current_context())
    return Location(
        API.mlirLocationFusedGet(context, length(locations), locations, metadata)
    )
end
