using Tangles
using Networks
using LinearAlgebra
using YaoBlocks
using DelegatorTraits
import DelegatorTraits: DelegatorTrait

@kwdef struct Circuit <: Tangles.AbstractTensorNetwork
    tn::Tangles.GenericTensorNetwork
    last_t::Dict{Site,Int} = Dict{Site,Int}()
end

DelegatorTrait(::Networks.Network, ::Circuit) = DelegateToField{:tn}()
DelegatorTrait(::Networks.Taggable, ::Circuit) = DelegateToField{:tn}()
DelegatorTrait(::Tangles.TensorNetwork, ::Circuit) = DelegateToField{:tn}()
DelegatorTrait(::Tangles.Pluggable, ::Circuit) = DelegateToField{:tn}()
DelegatorTrait(::Tangles.Lattice, ::Circuit) = DelegateToField{:tn}()

Base.copy(circ::Circuit) = Circuit(copy(circ.tn), Dict{Site,Int}())

function flatten_circuit(x)
    if any(i -> i isa ChainBlock, subblocks(x))
        flatten_circuit(YaoBlocks.Optimise.eliminate_nested(x))
    else
        x
    end
end

struct LaneAt{S}
    site::S
    t::Int
end

function Base.show(io::IO, lane::LaneAt)
    print(io, lane.site)
    return print(io, "@$(lane.t)")
end

moment(circuit::Circuit, site::Site) = circuit.last_t[site]

"""
    Convert a Yao circuit to a Circuit.
"""
function Base.convert(::Type{Circuit}, yaocirc::AbstractBlock)
    tn = GenericTensorNetwork()
    circuit = Circuit(tn, Dict{Site,Int}())

    for gate in flatten_circuit(yaocirc)
        # if gate isa Swap
        #     (a, b) = occupied_locs(gate)
        #     wire[a], wire[b] = wire[b], wire[a]
        #     continue
        # end

        gatesites = CartesianSite.(occupied_locs(gate))
        gateinds = Index.([Plug.(gatesites)..., Plug.(gatesites; isdual=true)...])

        # NOTE `YaoBlocks.mat` on m-site qubits still returns the operator on the full Hilbert space
        m = length(occupied_locs(gate))
        operator = if gate isa YaoBlocks.ControlBlock
            control((1:(m - 1))..., m => content(gate))(m)
        else
            content(gate)
        end

        array = reshape(collect(mat(operator)), fill(nlevel(operator), length(gateinds))...)

        addtensor!(circuit, Tensor(array, gateinds))
    end

    return circuit
end

function Tangles.addtensor!(circuit::Circuit, tensor::Tensor)
    target_plugs = plugs(tensor)
    target_plugs_in = filter(isdual, target_plugs)
    target_plugs_out = filter(!isdual, target_plugs)
    target_sites = unique!(site.(target_plugs))

    # if lane is not present, add an identity gate
    for plug in adjoint.(target_plugs_in)
        if !hasplug(circuit, plug)
            input, out = Index(LaneAt(site(plug), 1)), Index(LaneAt(site(plug), 2))
            addtensor!(circuit.tn, Tensor([1 0; 0 1], [input, out]))
            setplug!(circuit, input, plug')
            setplug!(circuit, out, plug)
            circuit.last_t[site(plug)] = 2
        end
    end

    # align gate tensor with the circuit
    tensor = replace(
        tensor,
        [
            Index(plug) => Index(LaneAt(site(plug), moment(circuit, site(plug)))) for
            plug in target_plugs_in
        ]...,
        [
            Index(plug) => Index(LaneAt(site(plug), moment(circuit, site(plug)) + 1)) for
            plug in target_plugs_out
        ]...,
    )

    addtensor!(circuit.tn, tensor)

    # update plug tags mapping
    for plug in target_plugs_out
        unsetplug!(circuit, plug)
        setplug!(circuit, Index(LaneAt(site(plug), moment(circuit, site(plug)) + 1)), plug)
    end

    # update the last_t for each site
    for site in target_sites
        circuit.last_t[site] = circuit.last_t[site] + 1
    end

    return circuit
end

"""
    Create an observable type Circuit.
"""
function create_observable(N)
    observable = chain(N, [put(i => Z) for i in 1:N]...) # Observable to measure: Z gate on each qubit

    # convert observable to circuit
    observable = convert(Circuit, observable)
    for tensor in tensors(observable)
        replace!(observable, tensor => Tensor(real(collect(parent(tensor))), inds(tensor)))
    end
    return observable
end
