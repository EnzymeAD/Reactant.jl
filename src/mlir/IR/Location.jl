struct Location
    ref::API.MlirLocation

    function Location(location)
        @assert !mlirIsNull(location) "cannot create Location with null MlirLocation"
        return new(location)
    end
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

Base.:(==)(a::Location, b::Location) = API.mlirLocationEqual(a, b)
context(location::Location) = Context(API.mlirLocationGetContext(location))

function Base.show(io::IO, location::Location)
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
