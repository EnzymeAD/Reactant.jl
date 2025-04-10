module Hostlists
module SlurmHostlists

import Libdl

const libslurm = Libdl.find_library(["libslurm"])
if !("" == libslurm)
    # We need to dlopen libslurm with RTLD_GLOBAL to make sure that all
    # dependencies are loaded correctly.
    Libdl.dlopen(libslurm, Libdl.RTLD_GLOBAL)
end

const hostlist_t = Ptr{Nothing}

slurm_hostlist_create(hostlist) = @ccall libslurm.slurm_hostlist_create(hostlist::Cstring)::hostlist_t
slurm_hostlist_count(hl::hostlist_t) = @ccall libslurm.slurm_hostlist_count(hl::hostlist_t)::Cint
slurm_hostlist_destroy(hl::hostlist_t) = @ccall libslurm.slurm_hostlist_destroy(hl::hostlist_t)::Cvoid
slurm_hostlist_find(hl::hostlist_t, hostname) = @ccall libslurm.slurm_hostlist_find(hl::hostlist_t, hostname::Cstring)::Cint
slurm_hostlist_push(hl::hostlist_t, hosts) = @ccall libslurm.slurm_hostlist_push(hl::hostlist_t,hosts::Cstring)::Cint
slurm_hostlist_push_host(hl::hostlist_t, host) = @ccall libslurm.slurm_hostlist_push_host(hl::hostlist_t, host::Cstring)::Cint
slurm_hostlist_ranged_string(hl::hostlist_t, n::Csize_t, buf) = @ccall libslurm.slurm_hostlist_ranged_string(hl::hostlist_t, n::Csize_t, buf::Ptr{UInt8})::Cssize_t
slurm_hostlist_shift(hl::hostlist_t) = @ccall libslurm.slurm_hostlist_shift(hl::hostlist_t)::Cstring
slurm_hostlist_uniq(hl::hostlist_t) = @ccall libslurm.slurm_hostlist_uniq(hl::hostlist_t)::Cvoid

mutable struct Hostlist
    hlist::hostlist_t
    
    function Hostlist(node_list::String)
        slurm_hl = slurm_hostlist_create(node_list)
        if slurm_hl == C_NULL
            error("Could not allocate memory for hostlist.")
        end
        hl = new(slurm_hl)
        finalizer(delete, hl)
        return hl
    end
end

export Hostlist

function delete(hl::Hostlist)
    slurm_hostlist_destroy(hl.hlist)
end

function Base.iterate(hl::Hostlist, state::Union{Nothing,Hostlist}=nothing) 
    hn_cstring = slurm_hostlist_shift(hl.hlist)
    (hn_cstring == C_NULL) && return nothing
    
    return unsafe_string(hn_cstring), hl 
end

Base.eltype(::Hostlist) = String
Base.IteratorEltype(::Type{Hostlist}) = Base.HasEltype()
Base.IteratorSize(::Type{Hostlist}) = Base.SizeUnknown()

function Base.convert(::Type{String}, hl::Hostlist, init_maxlen=8192)
    maxlen = init_maxlen
    hostnames = Vector{UInt8}(undef, maxlen)
    write_len = 0
    while true
        hostnames = Vector{UInt8}(undef, maxlen)
        write_len = slurm_hostlist_ranged_string(hl.hlist, UInt64(sizeof(hostnames)), hostnames)
        (write_len != -1) && break
        maxlen *= 2
    end
    hostlist = hostnames[1:write_len+1]
    hostlist[end] = 0 # ensure null-termination
    return GC.@preserve hostlist unsafe_string(pointer(hostlist))
end

Base.string(hl::Hostlist) = Base.convert(String, hl)
function Base.push!(x::Hostlist, y::String)
    slurm_hostlist_push(x.hlist, y)
    x
end
Base.length(x::Hostlist) = slurm_hostlist_count(x.hlist)

export string, push!, length

Base.show(io::IO, x::Hostlist) = print(io, string(x))

end

module SimpleHostlists

mutable struct Hostlist
    hlist::Vector{String}
    
    function Hostlist(node_list::String)
        hl = node_list |> x->split(x, ",") |> x->filter(!isempty, x) |> unique!
        return new(hl)
    end
end

Base.eltype(::Hostlist) = String
Base.IteratorEltype(::Type{Hostlist}) = Base.HasEltype()
Base.IteratorSize(::Type{Hostlist}) = Base.SizeUnknown()

Base.iterate(hl::Hostlist) = Base.iterate(hl.hlist)
Base.iterate(hl::Hostlist, state) = Base.iterate(hl.hlist, state) 

Base.convert(::Type{String}, hl::Hostlist) = join(hl.hlist, ",")
Base.string(hl::Hostlist) = Base.convert(String, hl)
function Base.push!(x::Hostlist, y::String)
    push!(x.hlist, y)
    x.hlist = x.hlist |> x->filter(!isempty, x) |> unique!
    x
end
Base.length(x::Hostlist) = length(x.hlist)

export string, push!, length

Base.show(io::IO, x::Hostlist) = print(io, string(x))

end

global Hostlist

function __init__()
    if "" == SlurmHostlists.libslurm
        @debug "libslurm.so not found, using SimpleHostlists"
        global const Hostlists.Hostlist = SimpleHostlists.Hostlist
    else
        @debug "libslurm.so found, using SlurmHostlists"
        global const Hostlists.Hostlist = SlurmHostlists.Hostlist
    end
end
end
