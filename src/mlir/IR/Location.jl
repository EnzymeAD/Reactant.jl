struct Location
    location::API.MlirLocation

    function Location(location)
        @assert !mlirIsNull(location) "cannot create Location with null MlirLocation"
        new(location)
    end
end

Location(; context::Context=context()) = Location(API.mlirLocationUnknownGet(context))

function Location(filename, line, column; context::Context=context())
    Location(API.mlirLocationFileLineColGet(context, filename, line, column))
end

function Location(callee::Location, caller::Location; context::Context=context())
    Location(API.mlirLocationCallSiteGet(context, callee, caller))
end

function Location(name::String, location::Location; context::Context=context())
    Location(API.mlirLocationNameGet(context, name, location))
end

# TODO rename to merge?
fuse(locations::Vector{Location}, metadata; context::Context=context()) = Location(API.mlirLocationFusedGet(context, length(locations), pointer(locations), metadata))

Base.convert(::Core.Type{API.MlirLocation}, location::Location) = location.location
Base.:(==)(a::Location, b::Location) = API.mlirLocationEqual(a, b)
context(location::Location) = Context(API.mlirLocationGetContext(location))

function Base.show(io::IO, location::Location)
    c_print_callback = @cfunction(print_callback, Cvoid, (API.MlirStringRef, Any))
    ref = Ref(io)
    print(io, "Location(#= ")
    API.mlirLocationPrint(location, c_print_callback, ref)
    print(io, " =#)")
end
