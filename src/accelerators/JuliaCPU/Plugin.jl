struct JuliaPlugin <: AbstractPlugin end

Client(::JuliaPlugin, ::Dict{String,Any}) = JuliaClient()

function attributes(::JuliaPlugin)
    callback = @cfunction(IR.print_callback, Cvoid, (API.MlirStringRef, Any))
    current, minimum = map((API.stablehloGetCurrentVersion, API.stablehloGetMinimumVersion)) do getversion
        io = IOBuffer()
        getversion(callback, Ref(io))
        parse.(Int64, split(String(take!(io)), '.'))
    end
    return ("stablehlo_current_version" => current, "stablehlo_minimum_version" => minimum)
end
