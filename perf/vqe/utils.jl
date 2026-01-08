using YaoBlocks
using Tangles

# simplified from `qiskit.circuit.library.efficient_su2`
function efficient_su2(nqubits, nlayers)
    gates = []

    for layer in 1:nlayers
        # apply a layer of SU(2) gates (i.e. single-qubit rotations that can generate the full SU(2) group)
        for qid in 1:nqubits
            # apply a single-qubit rotation gate
            push!(gates, put(qid => YaoBlocks.Ry(0.0)))
            push!(gates, put(qid => YaoBlocks.Rz(0.0)))
        end

        # apply a layer of entangling gates (i.e. two-qubit CX gates) = 
        for (qid_control, qid_target) in zip((nqubits - 1):-1:1, nqubits:-1:2)
            push!(gates, control(qid_control, qid_target => YaoBlocks.X))
        end
    end

    return chain(nqubits, gates...)
end

struct StackTag{T}
    id::T
    t::Int
end

function generic_stack(tns...)
    tn = GenericTensorNetwork()
    tns = copy.(tns)

    for (i, tni) in enumerate(tns)
        for ind in inds(tni)
            replace!(tni, ind => Index(StackTag(ind.tag, i)))
        end
    end

    append!(tn, all_tensors(tns[1]))
    for i in 2:length(tns)
        @align! outputs(tns[i - 1]) => inputs(tns[i])
        append!(tn, all_tensors(tns[i]))
    end

    return tn
end
