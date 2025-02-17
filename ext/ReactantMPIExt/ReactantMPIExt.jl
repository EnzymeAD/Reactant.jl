module ReactantMPIExt

using Reactant
using Reactant: MLIR
using MPI

function __init__()
    for name in (
        "MPI_Init",
        "MPI_Finalize",
        "MPI_Comm_rank",
        "MPI_Comm_size",
        "MPI_Send",
        "MPI_Recv",
        "MPI_Isend",
        "MPI_Irecv",
        "MPI_Barrier",
        "MPI_Wait",
        "MPI_Request_free",
    )
        sym = Libdl.dlsym(MPI.API.libmpi, name)
        @ccall MLIR.API.mlir_c.EnzymeJaXMapSymbol(name::Cstring, sym::Ptr{Cvoid})::Cvoid
    end
end

struct TracedRequest <: MPI.AbstractRequest
    mlir_data::Union{Nothing,MLIR.IR.Value}
end

include("Ops.jl")
include("Overrides.jl")

end # module
