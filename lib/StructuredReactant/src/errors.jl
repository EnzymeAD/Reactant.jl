"""
    FrontendError(message, [context])

Julia the frontend cannot express as a StableHLO program. `context` names the
methods being emitted, innermost first.
"""
struct FrontendError <: Exception
    message::String
    context::Vector{String}
end
FrontendError(message::AbstractString) = FrontendError(String(message), String[])

function Base.showerror(io::IO, err::FrontendError)
    print(io, "StructuredReactant: ", err.message)
    for frame in err.context
        print(io, "\n  in ", frame)
    end
end
