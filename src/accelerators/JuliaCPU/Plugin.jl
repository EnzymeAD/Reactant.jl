struct JuliaPlugin <: AbstractPlugin end

create_client(::JuliaPlugin, ::Dict{String,Any}) = JuliaClient()

function attributes(::JuliaPlugin)
    callback = @cfunction(IR.print_callback, Cvoid, (API.MlirStringRef, Any))
    current = let io = IOBuffer()
        API.stablehloGetCurrentVersion(callback, Ref(io))
        parse.(Int64, split(String(take!(io)), '.'))
    end
    minimum = let io = IOBuffer()
        API.stablehloGetMinimumVersion(callback, Ref(io))
        parse.(Int64, split(String(take!(io)), '.'))
    end
    return ("stablehlo_current_version" => current, "stablehlo_minimum_version" => minimum)
end
