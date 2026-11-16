struct HostPlugin <: AbstractPlugin end

Client(::HostPlugin, ::Dict{String,Any}) = HostClient()
