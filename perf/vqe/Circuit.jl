using Tangles
using Networks
using LinearAlgebra
using YaoBlocks
using DelegatorTraits
import DelegatorTraits: DelegatorTrait

struct Circuit <: Tangles.AbstractTensorNetwork
    tn::Tangles.GenericTensorNetwork
end

DelegatorTrait(::Networks.Network, ::Circuit) = DelegateToField{:tn}()
DelegatorTrait(::Networks.Taggable, ::Circuit) = DelegateToField{:tn}()
DelegatorTrait(::Tangles.TensorNetwork, ::Circuit) = DelegateToField{:tn}()
DelegatorTrait(::Tangles.Pluggable, ::Circuit) = DelegateToField{:tn}()
DelegatorTrait(::Tangles.Lattice, ::Circuit) = DelegateToField{:tn}()

Base.copy(circ::Circuit) = Circuit(copy(circ.tn))

function flatten_circuit(x)
    if any(i -> i isa ChainBlock, subblocks(x))
        flatten_circuit(YaoBlocks.Optimise.eliminate_nested(x))
    else
        x
    end
end

"""
    Convert a Yao circuit to a Circuit.
"""
function Base.convert(::Type{Circuit}, yaocirc::AbstractBlock)
    tn = GenericTensorNetwork()
    circuit = Circuit(tn)

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
            control((1:(m-1))..., m => content(gate))(m)
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

    for plug in filter(isdual, target_plugs) .|> adjoint
        if !hasplug(circuit, plug)
            input, out = Index(gensym(:tmp)), Index(gensym(:tmp))
            addtensor!(circuit.tn, Tensor([1 0; 0 1], [input, out]))
            setplug!(circuit, input, plug')
            setplug!(circuit, out, plug)
        end
    end

    tensor = replace(tensor, [Index(plug"i'") => ind_at(circuit, plug"i") for i in unique(site.(plugs(tensor)))]...)
    # for all the normal plugs in the operator
    # new_ind = Index(gensym(:tmp)) # Index((; layer=..., site=...))
    new_inds = Dict(plug"i" => Index(gensym(:tmp)) for i in unique(site.(plugs(tensor))))
    tensor = replace(tensor, [Index(k) => v for (k, v) in new_inds]...)

    addtensor!(circuit.tn, tensor)

    for (plug, new_ind) in new_inds
        unsetplug!(circuit, plug)
        setplug!(circuit, new_ind, plug)
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
